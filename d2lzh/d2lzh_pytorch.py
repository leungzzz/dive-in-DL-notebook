import collections
import math
import os
import random
import sys
import tarfile
import time
import zipfile
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn

from IPython import display
from matplotlib import pyplot as plt
# import mxnet as mx
# from mxnet import autograd, gluon, image, init, nd
# from mxnet.contrib import text
# from mxnet.gluon import data as gdata, loss as gloss, nn, utils as gutils
import numpy as np


VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']


VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]


def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    return plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0],
                         height=bbox[3]-bbox[1], fill=False, edgecolor=color,
                         linewidth=2)


class Benchmark():
    """Benchmark programs."""
    def __init__(self, prefix=None):
        self.prefix = prefix + ' ' if prefix else ''

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        print('%stime: %.4f sec' % (self.prefix, time.time() - self.start))


# def corr2d(X, K):
#     """Compute 2D cross-correlation."""
#     h, w = K.shape
#     Y = nd.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
#     for i in range(Y.shape[0]):
#         for j in range(Y.shape[1]):
#             Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
#     return Y
def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

def count_tokens(samples):
    """Count tokens in the data set."""
    token_counter = collections.Counter()
    for sample in samples:
        for token in sample:
            if token not in token_counter:
                token_counter[token] = 1
            else:
                token_counter[token] += 1
    return token_counter


def data_iter(batch_size, features, labels):
    """Iterate through a data set."""
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)


def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    """Sample mini-batches in a consecutive order from sequential data."""
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0 : batch_size * batch_len].reshape((
        batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i : i + num_steps]
        Y = indices[:, i + 1 : i + num_steps + 1]
        yield X, Y


def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    """Sample mini-batches in a random order from sequential data."""
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    def _data(pos):
        return corpus_indices[pos : pos + num_steps]

    for i in range(epoch_size):
        i = i * batch_size
        batch_indices = example_indices[i : i + batch_size]
        X = nd.array(
            [_data(j * num_steps) for j in batch_indices], ctx=ctx)
        Y = nd.array(
            [_data(j * num_steps + 1) for j in batch_indices], ctx=ctx)
        yield X, Y


def download_imdb(data_dir='../data'):
    """Download the IMDB data set for sentiment analysis."""
    url = ('http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
    sha1 = '01ada507287d82875905620988597833ad4e0903'
    fname = gutils.download(url, data_dir, sha1_hash=sha1)
    with tarfile.open(fname, 'r') as f:
        f.extractall(data_dir)


def _download_pikachu(data_dir):
    root_url = ('https://apache-mxnet.s3-accelerate.amazonaws.com/'
                'gluon/dataset/pikachu/')
    dataset = {'train.rec': 'e6bcb6ffba1ac04ff8a9b1115e650af56ee969c8',
               'train.idx': 'dcf7318b2602c06428b9988470c731621716c393',
               'val.rec': 'd6c33f799b4d058e82f2cb5bd9a976f69d72d520'}
    for k, v in dataset.items():
        gutils.download(root_url + k, os.path.join(data_dir, k), sha1_hash=v)


def download_voc_pascal(data_dir='../data'):
    """Download the Pascal VOC2012 Dataset."""
    voc_dir = os.path.join(data_dir, 'VOCdevkit/VOC2012')
    url = ('http://host.robots.ox.ac.uk/pascal/VOC/voc2012'
           '/VOCtrainval_11-May-2012.tar')
    sha1 = '4e443f8a2eca6b1dac8a6c57641b67dd40621a49'
    fname = gutils.download(url, data_dir, sha1_hash=sha1)
    with tarfile.open(fname, 'r') as f:
        f.extractall(data_dir)
    return voc_dir

'''
def evaluate_accuracy(data_iter, net, ctx=[mx.cpu()]):
    """Evaluate accuracy of a model on the given data set."""
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc_sum, n = nd.array([0]), 0
    for batch in data_iter:
        features, labels, _ = _get_batch(batch, ctx)
        for X, y in zip(features, labels):
            y = y.astype('float32')
            acc_sum += (net(X).argmax(axis=1) == y).sum().copyto(mx.cpu())
            n += y.size
        acc_sum.wait_to_read()
    return acc_sum.asscalar() / n
'''

# def evaluate_accuracy(data_iter, net):
#     acc_sum, n = 0.0, 0
#     for X, y in data_iter:
#         acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
#         n += y.shape[0]
#     return acc_sum / n


### 修改　精度估算函数使其支持cuda tensor
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device  # 无指定，则使用net中的设置

    acc_sum, num = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, nn.Module):  # net是nn.Module的子类
                net.eval()  # 进入评估模式
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()  # 回到训练模式
            else:  # net不是Module的子类，而是自己构造的模型
                if ('is_training' in net.__code__.co_varnames):
                    # 将这个参数设置为false
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            num += y.shape[0]
    return acc_sum / num


def _get_batch(batch, ctx):
    """Return features and labels on ctx."""
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx), features.shape[0])


# def get_data_ch7():
#     """Get the data set used in Chapter 7."""
#     data = np.genfromtxt('./airfoil_self_noise.dat', delimiter='\t')
#     data = (data - data.mean(axis=0)) / data.std(axis=0)
#     return nd.array(data[:, :-1]), nd.array(data[:, -1])
def get_data_ch7():
    data = np.genfromtxt('../Datasets/airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)

    return torch.tensor(data[:1500, :-1], dtype=torch.float32), \
           torch.tensor(data[:1500, -1], dtype=torch.float32)
    # 共1500个样本，每个样本5个特征


def get_fashion_mnist_labels(labels):
    """Get text label for fashion mnist."""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def get_tokenized_imdb(data):
    """Get the tokenized IMDB data set for sentiment analysis."""
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    return [tokenizer(review) for review, _ in data]


def get_vocab_imdb(data):
    """Get the vocab for the IMDB data set for sentiment analysis."""
    tokenized_data = get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return text.vocab.Vocabulary(counter, min_freq=5,
                                 reserved_tokens=['<pad>'])


def grad_clipping(params, theta, ctx):
    """Clip the gradient."""
    if theta is not None:
        norm = nd.array([0], ctx)
        for param in params:
            norm += (param.grad ** 2).sum()
        norm = norm.sqrt().asscalar()
        if norm > theta:
            for param in params:
                param.grad[:] *= theta / norm


# def linreg(X, w, b):
#     """Linear regression."""
#     return nd.dot(X, w) + b

def linreg(X, w, b):
    return torch.mm(X, w) + b

'''
def load_data_fashion_mnist(batch_size, resize=None, root=os.path.join(
        '~', '.mxnet', 'datasets', 'fashion-mnist')):
    """Download the fashion mnist dataset and then load into memory."""
    root = os.path.expanduser(root)
    transformer = []
    if resize:
        transformer += [gdata.vision.transforms.Resize(resize)]
    transformer += [gdata.vision.transforms.ToTensor()]
    transformer = gdata.vision.transforms.Compose(transformer)

    mnist_train = gdata.vision.FashionMNIST(root=root, train=True)
    mnist_test = gdata.vision.FashionMNIST(root=root, train=False)
    num_workers = 0 if sys.platform.startswith('win32') else 4

    train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                                  batch_size, shuffle=True,
                                  num_workers=num_workers)
    test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                                 batch_size, shuffle=False,
                                 num_workers=num_workers)
    return train_iter, test_iter
'''

# def load_data_fashion_mnist(batch_size, resize=None, root=os.path.join('.', 'Datasets', 'FashionMINST')):
#     """Download the fashion mnist dataset and then load into memory."""
#
#     mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=False, transform=transforms.ToTensor())
#     mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=False, transform=transforms.ToTensor())
#
#     batch_size = 256
#
#     if sys.platform.startswith("win"):
#         num_workers = 0  #
#     else:
#         num_workers = 4  # 表示如果在其他平台，则使用4个进程来读取数据
#
#     # 以迭代的方式进行数据读取
#     train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#
#     return train_iter, test_iter

def load_data_fashion_mnist(batch_size, resize=None, root='../Datasets/FashionMINST/'):
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(resize))  # 变换１
    trans.append(torchvision.transforms.ToTensor())  # 变换２

    transform = torchvision.transforms.Compose(trans)  # 组合１/2等多种变换
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=False, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=True, download=False, transform=transform)

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_iter, test_iter

        
def load_data_jay_lyrics():
    """Load the Jay Chou lyric data set (available in the Chinese book)."""
    with zipfile.ZipFile('../data/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size


def load_data_pikachu(batch_size, edge_size=256):
    """Download the pikachu dataest and then load into memory."""
    data_dir = '../data/pikachu'
    _download_pikachu(data_dir)
    train_iter = image.ImageDetIter(
        path_imgrec=os.path.join(data_dir, 'train.rec'),
        path_imgidx=os.path.join(data_dir, 'train.idx'),
        batch_size=batch_size,
        data_shape=(3, edge_size, edge_size),
        shuffle=True,
        rand_crop=1,
        min_object_covered=0.95,
        max_attempts=200)
    val_iter = image.ImageDetIter(
        path_imgrec=os.path.join(data_dir, 'val.rec'),
        batch_size=batch_size,
        data_shape=(3, edge_size, edge_size),
        shuffle=False)
    return train_iter, val_iter


def load_data_time_machine():
    """Load the time machine data set (available in the English book)."""
    with open('../data/timemachine.txt') as f:
        corpus_chars = f.read()
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ').lower()
    corpus_chars = corpus_chars[0:10000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size


def _make_list(obj, default_values=None):
    if obj is None:
        obj = default_values
    elif not isinstance(obj, (list, tuple)):
        obj = [obj]
    return obj


def mkdir_if_not_exist(path):
    """Make a directory if it does not exist."""
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))


def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx):
    """Predict next chars with a RNN model"""
    state = init_rnn_state(1, num_hiddens, ctx)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = to_onehot(nd.array([output[-1]], ctx=ctx), vocab_size)
        (Y, state) = rnn(X, state, params)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[i] for i in output])


def predict_rnn_gluon(prefix, num_chars, model, vocab_size, ctx, idx_to_char,
                      char_to_idx):
    """Precit next chars with a Gluon RNN model"""
    state = model.begin_state(batch_size=1, ctx=ctx)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = nd.array([output[-1]], ctx=ctx).reshape((1, 1))
        (Y, state) = model(X, state)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[i] for i in output])


def predict_sentiment(net, vocab, sentence):
    """Predict the sentiment of a given sentence."""
    sentence = nd.array(vocab.to_indices(sentence), ctx=try_gpu())
    label = nd.argmax(net(sentence.reshape((1, -1))), axis=1)
    return 'positive' if label.asscalar() == 1 else 'negative'


def preprocess_imdb(data, vocab):
    """Preprocess the IMDB data set for sentiment analysis."""
    max_l = 500

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [
            vocab.token_to_idx['<pad>']] * (max_l - len(x))

    tokenized_data = get_tokenized_imdb(data)
    features = nd.array([pad(vocab.to_indices(x)) for x in tokenized_data])
    labels = nd.array([score for _, score in data])
    return features, labels


def read_imdb(folder='train'):
    """Read the IMDB data set for sentiment analysis."""
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join('../data/aclImdb/', folder, label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)
    return data


def read_voc_images(root='../data/VOCdevkit/VOC2012', is_train=True):
    """Read VOC images."""
    txt_fname = '%s/ImageSets/Segmentation/%s' % (
        root, 'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [None] * len(images), [None] * len(images)
    for i, fname in enumerate(images):
        features[i] = image.imread('%s/JPEGImages/%s.jpg' % (root, fname))
        labels[i] = image.imread(
            '%s/SegmentationClass/%s.png' % (root, fname))
    return features, labels

'''
class Residual(nn.Block):
    """The residual block."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = nd.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return nd.relu(Y + X)
'''

# def resnet18(num_classes):
#     """The ResNet-18 model."""
#     net = nn.Sequential()
#     net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
#             nn.BatchNorm(), nn.Activation('relu'))
#
#     def resnet_block(num_channels, num_residuals, first_block=False):
#         blk = nn.Sequential()
#         for i in range(num_residuals):
#             if i == 0 and not first_block:
#                 blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
#             else:
#                 blk.add(Residual(num_channels))
#         return blk
#
#     net.add(resnet_block(64, 2, first_block=True),
#             resnet_block(128, 2),
#             resnet_block(256, 2),
#             resnet_block(512, 2))
#     net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
#     return net

# ########################### 5.11 ################################
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

def resnet18(output=10, in_channels=3):
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", GlobalAvgPool2d())  # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
    net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(512, output)))
    return net


def resnet18_cifar10(output=10, in_channels=3):
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("global_avg_pool", GlobalAvgPool2d())  # GlobalAvgPool2d的输出: (Batch, 256, 1, 1)
    net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(256, output)))
    return net


class GlobalAvgPool2d(nn.Module): # 全局平均池化层
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d( x, kernel_size=x.size()[2:] )


'''
class RNNModel(nn.Block):
    """RNN model."""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)

    def forward(self, inputs, state):
        X = nd.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(X, state)
        output = self.dense(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
'''

def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    """Plot x and log(y)."""
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()


def set_figsize(figsize=(3.5, 2.5)):
    """Set matplotlib figure size."""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def sgd(params, lr, batch_size):
    """Mini-batch stochastic gradient descent."""
    for param in params:
        param[:] = param - lr * param.grad / batch_size


# def show_bboxes(axes, bboxes, labels=None, colors=None):
#     """Show bounding boxes."""
#     labels = _make_list(labels)
#     colors = _make_list(colors, ['b', 'g', 'r', 'm', 'k'])
#     for i, bbox in enumerate(bboxes):
#         color = colors[i % len(colors)]
#         rect = bbox_to_rect(bbox.asnumpy(), color)
#         axes.add_patch(rect)
#         if labels and len(labels) > i:
#             text_color = 'k' if color == 'w' else 'w'
#             axes.text(rect.xy[0], rect.xy[1], labels[i],
#                       va='center', ha='center', fontsize=9, color=text_color,
#                       bbox=dict(facecolor=color, lw=0))

def show_bboxes(axes, bboxes, labels=None, colors=None):
    """
        描绘以某个像素为中心的所有锚框在input上
    """
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox.detach().cpu().numpy(), color)
        axes.add_patch(rect)  # 先画出一个框
        if labels and len(labels) > i:  # 标注文字、颜色等
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0],  # 位置x
                      rect.xy[1],  # 位置y
                      labels[i],  # 文字
                      va='center',  #
                      ha='center',
                      fontsize=6,
                      color=text_color,
                      bbox=dict(facecolor=color, lw=0)
                      )


############# Chap 9 ################
def MultiBoxPrior(feature_map, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5]):
    '''
    Args:
        feature_map: torch tensor, Shape: [N, C, H, W].
        sizes: List of sizes (0~1)
        ratios: List of aspect ratios (non-negative)
    Returns:
        anchors of shape (1, num_anchors, 4). # batch, num_anc, positions
    '''
    # s与r的组合
    pairs = []
    for r in ratios:
        pairs.append([sizes[0], math.sqrt(r)])  # 首先保证有s_1
    for s in sizes[1:]:
        pairs.append([s, math.sqrt(ratios[0])])  # 其次要保证有r_1

    pairs = np.array(pairs)
    #     print(pairs)

    ss1 = pairs[:, 0] * pairs[:, 1]  # size * sqrt(ratio)
    ss2 = pairs[:, 0] / pairs[:, 1]  # size / sqrt(ratio)
    #     print(ss1)
    #     print(ss2)

    base_anchors = np.stack([-ss1, -ss2, ss1, ss2], axis=1) / 2
    # print(base_anchors)

    h, w = feature_map.shape[-2:]
    shifts_x = np.arange(0, w) / w
    #     print(shifts_x)
    shifts_y = np.arange(0, h) / h
    shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)
    shift_x = shift_x.reshape(-1)
    #     print(shift_x.shape)
    shift_y = shift_y.reshape(-1)
    #     print(shift_y.shape)  # (69069,)
    shifts = np.stack((shift_x, shift_y, shift_x, shift_y), axis=1)

    anchors = shifts.reshape((-1, 1, 4)) + base_anchors.reshape((1, -1, 4))

    return torch.tensor(anchors, dtype=torch.float32).view(1, -1, 4)


def compute_intersection(set_1, set_2):
    """
    计算anchor之间的交集
    Args:
        set_1: a tensor of dimensions (n1, 4), anchor表示成(xmin, ymin, xmax, ymax)
        set_2: a tensor of dimensions (n2, 4), anchor表示成(xmin, ymin, xmax, ymax)
    Returns:
        intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, shape: (n1, n2)
    """
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1),
                             set_2[:, :2].unsqueeze(0))  # (n1,n2,2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1),
                             set_2[:, 2:].unsqueeze(0))  # (n1,n2,2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1,n2,2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1,n2)


def compute_jaccard(set_1, set_2):
    """
    计算anchor之间的Jaccard系数(IoU)
    Args:
        set_1: a tensor of dimensions (n1, 4), anchor表示成(xmin, ymin, xmax, ymax)
        set_2: a tensor of dimensions (n2, 4), anchor表示成(xmin, ymin, xmax, ymax)
    Returns:
        Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, shape: (n1, n2)
    """
    # find intersections
    intersection = compute_intersection(set_1, set_2)  # (n1, n2)
    print(intersection.shape)

    # find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # find the union
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


def assign_anchor(bb, anchor, jaccard_threshold=0.5):
    """
    为每个anchor分配真实的bb,依据是jaccard系数/iou
    Args:
        bb: 真实边界框(bounding box), shape:（nb, 4）
        anchor: 待分配的anchor, shape:（na, 4）
        jaccard_threshold: 预先设定的阈值
    Returns:
        assigned_idx: shape: (na, ), 每个anchor分配的真实bb对应的索引, 若未分配任何bb则为-1
    """
    na = anchor.shape[0]
    nb = bb.shape[0]
    jaccard = compute_jaccard(anchor, bb).detach().cpu().numpy()  # R^{anchor x bb}
    assigned_idx = np.ones(na) * -1  # 为每一个anchor分配一个bb的id,初始值-1

    # 先为每个bb分配一个anchor
    jaccard_cp = jaccard.copy()
    for j in range(nb):
        i = np.argmax(jaccard_cp[:, j])  # 第j个真实边界框索引找到最大的jaccard
        assigned_idx[i] = j
        jaccard_cp[i, :] = float("-inf")  # 相当于永远不会再索引该行（因为已经分配完毕）

    # 对其他未得到分配的anchor来说，再次分配，需要考虑jaccard_threshold
    for i in range(na):
        if assigned_idx[i] == -1:  # 通过索引该数组确定jaccard矩阵内的分配情况
            j = np.argmax(jaccard[i, :])
            if jaccard[i, j] >= jaccard_threshold:
                assigned_idx[i] = j

    return torch.tensor(assigned_idx, dtype=torch.long)


def xy_to_cxcy(xy):
    """
     将(x_min, y_min, x_max, y_max)形式的anchor转换成(center_x, center_y, w, h)形式的.
    Args:
        xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    Returns:
        bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)

    """

    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # center_x, center_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def MultiBoxTarget(anchor, label):
    """
    Function:
        为anchor分配真实的label. 相比于assign_anchor函数，这里的anchor和label信息更复杂。
    Args:
        anchor: torch tensor, 输入的锚框, 一般是通过MultiBoxPrior生成, shape:（1，锚框总数，4）
        label: 真实标签, shape为(bn, 每张图片最多的真实锚框数, 5)
               第二维中，如果给定图片没有这么多锚框, 可以先用-1填充空白, 最后一维中的元素为[类别标签, 四个坐标值]
    Returns:
        列表, [bbox_offset, bbox_mask, cls_labels]
        bbox_offset: 每个锚框的标注偏移量，形状为(bn，锚框总数*4)
        bbox_mask: 形状同bbox_offset, 每个锚框的掩码, 一一对应上面的偏移量, 负类锚框(背景)对应的掩码均为0, 正类锚框的掩码均为1
        cls_labels: 每个锚框的标注类别, 其中0表示为背景, 形状为(bn，锚框总数)
    """

    assert len(anchor.shape) == 3 and len(label.shape) == 3
    bn = label.shape[0]

    def MultiBoxTarget_one(anc, lab, eps=1e-6):
        """
        Function:
            MultiBoxTarget函数的辅助函数, 处理batch中的一个
            给定ancs, 给定label, 根据 assign_anchor函数所给定的索引 来计算每个anc的类别、偏移量和mask
        Args:
            anc: shape of (锚框总数, 4)
            lab: shape of (真实锚框数, 5), 5代表[类别标签, 四个坐标值]
            eps: 一个极小值, 防止log0
        Returns:
            offset: (锚框总数*4, )
            bbox_mask: (锚框总数*4, ), 0代表背景, 1代表非背景
            cls_labels: (锚框总数, 4), 0代表背景
        """
        an = anc.shape[0]

        assigned_idx = assign_anchor(lab[:, 1:], anc)  # (锚框总数, )
        bbox_mask = ((assigned_idx >= 0).float().unsqueeze(-1)).repeat(1, 4)  # (锚框总数, 4)
        cls_labels = torch.zeros(an, dtype=torch.long)
        assigned_bb = torch.zeros((an, 4), dtype=torch.float32)

        for i in range(an):
            bb_idx = assigned_idx[i]
            if bb_idx >= 0:
                cls_labels[i] = lab[bb_idx, 0].long().item() + 1
                assigned_bb[i, :] = lab[bb_idx, 1:]

        center_anc = xy_to_cxcy(anc)
        center_assigned_anc = xy_to_cxcy(assigned_bb)

        offset_xy = 10.0 * (center_assigned_anc[:, :2] - center_anc[:, :2]) / center_anc[:, 2:]
        offset_wh = 5.0 * (center_assigned_anc[:, 2:] - center_anc[:, 2:]) / center_anc[:, 2:]
        offset = torch.cat([offset_xy, offset_wh], dim=1) * bbox_mask

        return offset.view(-1), bbox_mask.view(-1), cls_labels

    batch_offset = []
    batch_mask = []
    batch_cls_labels = []
    for b in range(bn):
        offset, bbox_mask, cls_label = MultiBoxTarget_one(anchor[0, :, :], label[b, :, :])

        batch_offset.append(offset)
        batch_mask.append(bbox_mask)
        batch_cls_labels.append(cls_label)

    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    cls_labels = torch.stack(batch_cls_labels)

    return [bbox_offset, bbox_mask, cls_labels]


from collections import namedtuple

Pred_BB_Tuple = namedtuple("Pred_BB_Tuple", ["index", 'class_id', 'confidence', 'xyxy'])


def non_max_suppression(bb_info_list, nms_threshold=0.5):
    """
    函数：给定预测边界框，将相似的预测边界框剔除
    Args:
        bb_info_list: Pred_BB_Tuple列表，包含类别，置信度，坐标值等信息
        nms_threshold: 每循环一次，小于该值的bb得以保留做下一轮循环

    Output：
        output: 经过筛选后的预测边界框
    """

    output = []
    # 根据置信度大小，排序
    sorted_bb_info_list = sorted(bb_info_list, key=lambda x: x.confidence, reverse=True)

    while (len(sorted_bb_info_list) != 0):
        best = sorted_bb_info_list.pop(0)  # 拿出第一个置信度最高的bb_list
        output.append(best)

        if len(sorted_bb_info_list) == 0:
            break

        # 为计算iou做准备
        bb_xyxy = []
        for bb in sorted_bb_info_list:
            bb_xyxy.append(bb.xyxy)

        # 计算iou, 计算预测边界框之间的交并比,为剔除做准备
        iou = compute_jaccard(torch.tensor([best.xyxy]), torch.tensor(bb_xyxy))[0]

        # 根据计算得到的iou进行非极大值抑制
        n = len(sorted_bb_info_list)
        sorted_bb_info_list = [sorted_bb_info_list[i] for i in range(n) if iou[i] <= nms_threshold]

    return output


def MultiBoxDetection(cls_prob, loc_pred, anchor, nms_threshold=0.5):
    """
    Function:
        将原生锚框和预测概率+预测偏移量进行整合，输出的是经过系统认证的（预测框+预测类别+预测置信度等信息）
    Args:
        cls_prob: 经过softmax后得到的各个锚框的类别预测概率, shape:(bn, 预测总类别数+1, 锚框个数)
        loc_pred: 预测的各个锚框的偏移量, shape:(bn, 锚框个数*4)
        anchor: MultiBoxPrior输出的默认锚框, shape: (1, 锚框个数, 4),anchor表示成归一化(xmin, ymin, xmax, ymax).
        nms_threshold: 非极大抑制中的阈值
    Returns:
        所有锚框的信息, shape: (bn, 锚框个数, 6)
        每个锚框信息由[class_id, confidence, xmin, ymin, xmax, ymax]表示
        class_id=-1 表示背景或在非极大值抑制中被移除了
    """
    assert len(cls_prob.shape) == 3 and len(loc_pred.shape) == 2 and len(anchor.shape) == 3

    bn = cls_prob.shape[0]

    def MultiBoxDetection_one(c_p, l_p, anc, nms_threshold=0.5):
        """
        MultiBoxDetection的辅助函数, 处理batch中的一个
        Args:
            c_p: (预测总类别数+1, 锚框个数)
            l_p: (锚框个数*4, )
            anc: (锚框个数, 4)
            nms_threshold: 非极大抑制中的阈值
        Return:
            output: (锚框个数, 6)
        """

        pred_bb_num = c_p.shape[1]
        anc = (anc + l_p.view(pred_bb_num, 4)).detach().cpu().numpy()

        confidence, class_id = torch.max(c_p, 0)  # (1, 锚框个数)
        confidence = confidence.detach().cpu().numpy()
        class_id = class_id.detach().cpu().numpy()

        pred_bb_info = [Pred_BB_Tuple(
            index=i,
            class_id=class_id[i] - 1,
            confidence=confidence[i],
            xyxy=[*anc[i]]) for i in range(pred_bb_num)]
        # 正类的index
        obj_bb_idx = [bb.index for bb in non_max_suppression(pred_bb_info, nms_threshold)]

        output = []  # 容纳筛选后的预测边界框
        for bb in pred_bb_info:
            output.append([
                (bb.class_id if bb.index in obj_bb_idx else -1.0),
                bb.confidence,
                *bb.xyxy
            ])

        return torch.tensor(output)  # shape: (锚框个数， 6)

    batch_output = []
    for b in range(bn):
        batch_output.append(MultiBoxDetection_one(cls_prob[b], loc_pred[b], anchor[0], nms_threshold))

    return torch.stack(batch_output)


# def show_fashion_mnist(images, labels):
#     """Plot Fashion-MNIST images with labels."""
#     use_svg_display()
#     _, figs = plt.subplots(1, len(images), figsize=(12, 12))
#     for f, img, lbl in zip(figs, images, labels):
#         f.imshow(img.reshape((28, 28)).asnumpy())
#         f.set_title(lbl)
#         f.axes.get_xaxis().set_visible(False)
#         f.axes.get_yaxis().set_visible(False)

def show_fashion_mnist(images, labels):
    use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


# def show_images(imgs, num_rows, num_cols, scale=2):
#     """Plot a list of images."""
#     figsize = (num_cols * scale, num_rows * scale)
#     _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
#     for i in range(num_rows):
#         for j in range(num_cols):
#             axes[i][j].imshow(imgs[i * num_cols + j].asnumpy())
#             axes[i][j].axes.get_xaxis().set_visible(False)
#             axes[i][j].axes.get_yaxis().set_visible(False)
#     return axes

def show_images(imgs, num_rows, num_cols, scale=2):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    plt.show()
    return axes


def show_trace_2d(f, res):
    """Show the trace of 2d variables during optimization."""
    x1, x2 = zip(*res)
    set_figsize()
    plt.plot(x1, x2, '-o', color='#ff7f0e')
    x1 = np.arange(-5.5, 1.0, 0.1)
    x2 = np.arange(min(-3.0, min(x2) - 1), max(1.0, max(x2) + 1), 0.1)
    x1, x2 = np.meshgrid(x1, x2)
    plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


# def squared_loss(y_hat, y):
#     """Squared loss."""
#     return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def squared_loss(y_hat, y):
    # 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
    return ((y_hat - y.view(y_hat.size())) ** 2) / 2


def to_onehot(X, size):
    """Represent inputs with one-hot encoding."""
    return [nd.one_hot(x, size) for x in X.T]


# def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs):
#     """Train and evaluate a model."""
#     print('training on', ctx)
#     if isinstance(ctx, mx.Context):
#         ctx = [ctx]
#     for epoch in range(num_epochs):
#         train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()
#         for i, batch in enumerate(train_iter):
#             Xs, ys, batch_size = _get_batch(batch, ctx)
#             with autograd.record():
#                 y_hats = [net(X) for X in Xs]
#                 ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
#             for l in ls:
#                 l.backward()
#             trainer.step(batch_size)
#             train_l_sum += sum([l.sum().asscalar() for l in ls])
#             n += sum([l.size for l in ls])
#             train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar()
#                                  for y_hat, y in zip(y_hats, ys)])
#             m += sum([y.size for y in ys])
#         test_acc = evaluate_accuracy(test_iter, net, ctx)
#         print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
#               'time %.1f sec'
#               % (epoch + 1, train_l_sum / n, train_acc_sum / m, test_acc,
#                  time.time() - start))
def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


def train_2d(trainer):
    """Optimize the objective function of 2d variables with a customized trainer."""
    x1, x2 = -5, -2
    s_x1, s_x2 = 0, 0
    res = [(x1, x2)]
    for i in range(20):
        x1, x2, s_x1, s_x2 = trainer(x1, x2, s_x1, s_x2)
        res.append((x1, x2))
    print('epoch %d, x1 %f, x2 %f' % (i+1, x1, x2))
    return res


def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, ctx, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    """Train an RNN model and predict the next item in the sequence."""
    if is_random_iter:
        data_iter_fn = data_iter_random
    else:
        data_iter_fn = data_iter_consecutive
    params = get_params()
    loss = gloss.SoftmaxCrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:
            state = init_rnn_state(batch_size, num_hiddens, ctx)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx)
        for X, Y in data_iter:
            if is_random_iter:
                state = init_rnn_state(batch_size, num_hiddens, ctx)
            else:
                for s in state:
                    s.detach()
            with autograd.record():
                inputs = to_onehot(X, vocab_size)
                (outputs, state) = rnn(inputs, state, params)
                outputs = nd.concat(*outputs, dim=0)
                y = Y.T.reshape((-1,))
                l = loss(outputs, y).mean()
            l.backward()
            grad_clipping(params, clipping_theta, ctx)
            sgd(params, lr, 1)
            l_sum += l.asscalar() * y.size
            n += y.size

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(
                    prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx))


def train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes):
    """Train an Gluon RNN model and predict the next item in the sequence."""
    loss = gloss.SoftmaxCrossEntropyLoss()
    model.initialize(ctx=ctx, force_reinit=True, init=init.Normal(0.01))
    trainer = gluon.Trainer(model.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0, 'wd': 0})

    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_consecutive(
            corpus_indices, batch_size, num_steps, ctx)
        state = model.begin_state(batch_size=batch_size, ctx=ctx)
        for X, Y in data_iter:
            for s in state:
                s.detach()
            with autograd.record():
                (output, state) = model(X, state)
                y = Y.T.reshape((-1,))
                l = loss(output, y).mean()
            l.backward()
            params = [p.data() for p in model.collect_params().values()]
            grad_clipping(params, clipping_theta, ctx)
            trainer.step(1)
            l_sum += l.asscalar() * y.size
            n += y.size

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn_gluon(
                    prefix, pred_len, model, vocab_size, ctx, idx_to_char,
                    char_to_idx))


# def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
#               params=None, lr=None, trainer=None):
#     """Train and evaluate a model with CPU."""
#     for epoch in range(num_epochs):
#         train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
#         for X, y in train_iter:
#             with autograd.record():
#                 y_hat = net(X)
#                 l = loss(y_hat, y).sum()
#             l.backward()
#             if trainer is None:
#                 sgd(params, lr, batch_size)
#             else:
#                 trainer.step(batch_size)
#             y = y.astype('float32')
#             train_l_sum += l.asscalar()
#             train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
#             n += y.size
#         test_acc = evaluate_accuracy(test_iter, net)
#         print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
#               % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


# ########################### 3.7 #####################################3
class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


# def train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx,
#               num_epochs):
#     """Train and evaluate a model with CPU or GPU."""
#     print('training on', ctx)
#     loss = gloss.SoftmaxCrossEntropyLoss()
#     for epoch in range(num_epochs):
#         train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
#         for X, y in train_iter:
#             X, y = X.as_in_context(ctx), y.as_in_context(ctx)
#             with autograd.record():
#                 y_hat = net(X)
#                 l = loss(y_hat, y).sum()
#             l.backward()
#             trainer.step(batch_size)
#             y = y.astype('float32')
#             train_l_sum += l.asscalar()
#             train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
#             n += y.size
#         test_acc = evaluate_accuracy(test_iter, net, ctx)
#         print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
#               'time %.1f sec'
#               % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc,
#                  time.time() - start))


def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)  # 网络模型
    print('training on ', device)
    loss = torch.nn.CrossEntropyLoss()  # 损失函数
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n, batch_count, start_time = 0.0, 0.0, 0, 0, time.time()

        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            ls = loss(y_hat, y)
            optimizer.zero_grad()
            ls.backward()
            optimizer.step()

            train_loss_sum += ls.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
            batch_count += 1

        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train_acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_loss_sum / batch_count,
                 train_acc_sum / n, test_acc, time.time() - start_time))


# def train_ch7(trainer_fn, states, hyperparams, features, labels, batch_size=10,
#               num_epochs=2):
#     """Train a linear regression model."""
#     net, loss = linreg, squared_loss
#     w, b = nd.random.normal(scale=0.01, shape=(features.shape[1], 1)), nd.zeros(1)
#     w.attach_grad()
#     b.attach_grad()
#
#     def eval_loss():
#         return loss(net(features, w, b), labels).mean().asscalar()
#
#     ls = [eval_loss()]
#     data_iter = gdata.DataLoader(
#         gdata.ArrayDataset(features, labels), batch_size, shuffle=True)
#     for _ in range(num_epochs):
#         start = time.time()
#         for batch_i, (X, y) in enumerate(data_iter):
#             with autograd.record():
#                 l = loss(net(X, w, b), y).mean()
#             l.backward()
#             trainer_fn([w, b], states, hyperparams)
#             if (batch_i + 1) * batch_size % 100 == 0:
#                 ls.append(eval_loss())
#     print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
#     set_figsize()
#     plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def train_ch7(optimizer_fn, states, hyperparams, features, labels,
              batch_size=10, num_epochs=2):
    # 初始化模型
    net, loss = linreg, squared_loss

    w = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(features.shape[1], 1)), dtype=torch.float32),
                           requires_grad=True)
    b = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)

    def eval_loss():
        return loss(net(features, w, b), labels).mean().item()

    ls = [eval_loss()]
    data_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, labels), batch_size, shuffle=True)

    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            l = loss(net(X, w, b), y).mean()  # 使用平均损失

            # 梯度清零
            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()

            l.backward()
            optimizer_fn([w, b], states, hyperparams)  # 迭代模型参数
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())  # 每100个样本记录下当前训练误差
    # 打印结果和作图
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    set_figsize()
    plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


def train_pytorch_ch7(optimizer_fn, optimizer_hyperparams,
                      features, labels, batch_size=10, num_epochs=2):
    net = nn.Sequential(
        nn.Linear(features.shape[-1], 1)
    )

    loss = nn.MSELoss()
    optimizer = optimizer_fn(net.parameters(), **optimizer_hyperparams)

    def eval_loss():
        return loss(net(features).view(-1), labels).item() / 2

    ls = [eval_loss()]
    data_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, labels),
        batch_size, shuffle=True,
    )

    for _ in range(num_epochs):
        start_time = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            l = loss(net(X).view(-1), y) / 2
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())

    # 打印结果和作图
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start_time))
    set_figsize()
    plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

def train_gluon_ch7(trainer_name, trainer_hyperparams, features, labels,
                    batch_size=10, num_epochs=2):
    """Train a linear regression model with a given Gluon trainer."""
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    loss = gloss.L2Loss()

    def eval_loss():
        return loss(net(features), labels).mean().asscalar()

    ls = [eval_loss()]
    data_iter = gdata.DataLoader(
        gdata.ArrayDataset(features, labels), batch_size, shuffle=True)
    trainer = gluon.Trainer(net.collect_params(),
                            trainer_name, trainer_hyperparams)
    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    set_figsize()
    plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')


def try_all_gpus():
    """Return all available GPUs, or [mx.cpu()] if there is no GPU."""
    ctxes = []
    try:
        for i in range(16):
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx=ctx)
            ctxes.append(ctx)
    except mx.base.MXNetError:
        pass
    if not ctxes:
        ctxes = [mx.cpu()]
    return ctxes


def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()."""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx


def use_svg_display():
    """Use svg format to display plot in jupyter"""
    display.set_matplotlib_formats('svg')


def voc_label_indices(colormap, colormap2label):
    """Assign label indices for Pascal VOC2012 Dataset."""
    colormap = colormap.astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]


def voc_rand_crop(feature, label, height, width):
    """Random cropping for images of the Pascal VOC2012 Dataset."""
    feature, rect = image.random_crop(feature, (width, height))
    label = image.fixed_crop(label, *rect)
    return feature, label

'''
class VOCSegDataset(gdata.Dataset):
    """The Pascal VOC2012 Dataset."""
    def __init__(self, is_train, crop_size, voc_dir, colormap2label):
        self.rgb_mean = nd.array([0.485, 0.456, 0.406])
        self.rgb_std = nd.array([0.229, 0.224, 0.225])
        self.crop_size = crop_size
        data, labels = read_voc_images(root=voc_dir, is_train=is_train)
        self.data = [self.normalize_image(im) for im in self.filter(data)]
        self.labels = self.filter(labels)
        self.colormap2label = colormap2label
        print('read ' + str(len(self.data)) + ' examples')

    def normalize_image(self, data):
        return (data.astype('float32') / 255 - self.rgb_mean) / self.rgb_std

    def filter(self, images):
        return [im for im in images if (
            im.shape[0] >= self.crop_size[0] and
            im.shape[1] >= self.crop_size[1])]

    def __getitem__(self, idx):
        data, labels = voc_rand_crop(self.data[idx], self.labels[idx],
                                     *self.crop_size)
        return (data.transpose((2, 0, 1)),
                voc_label_indices(labels, self.colormap2label))

    def __len__(self):
        return len(self.data)
'''
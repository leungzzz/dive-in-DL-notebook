# 进入测试环节（用所有训练集训练，所有的测试集进行测试）
import collections
import math

import os
import shutil
import time
import zipfile

import sys
sys.path.append('../d2lzh/')
import d2lzh_pytorch as d2l

import torch
import torchvision
from torchvision import models
from torch import nn, optim
from torchvision.datasets import ImageFolder as IF
from torch.utils.data import DataLoader as DL

demo = False
data_dir = '/home/tpg/Datasets/az_train_val/'
test_dir = "test/"
PATH = "./pretrained_resnet18_my.pth"
batch_size = 32

#### 准备数据集
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])
# 封装
test_data_path = os.path.join(data_dir, test_dir)
test_data = IF(test_data_path, transform=transform_test)
test_iter = DL(test_data, batch_size, shuffle=False)

#### 加载模型
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, 2)
model.load_state_dict(torch.load(PATH))
# print(model.state_dict())

#### 预测结果
preds = []
for X, _ in test_iter:
    y_hat = model(X)
    preds.extend(y_hat.cpu())  # 将每个样本的预测值计算出来。预测结果如 0,1
print(preds[0])

# #### 映射到具体类
# ids = sorted(os.listdir(os.path.join(data_dir, test_dir)))
# with open("/home/tpg/Datasets/az_train_val/test/submission.csv", 'w') as f:
#     f.write('id,' + ",".join(train_data.classes) + '\n')
#     for i, output in zip(ids, preds):
#         f.write(
#             i.split('.')[0] + ',' + ','.join([str(num)
#                                               for num in output]) + '\n')
from PIL import Image
import numpy as np
import math
import torch

import sys
import matplotlib.pyplot as plt
sys.path.append('./d2lzh/')
import d2lzh_pytorch as d2l
print(torch.__version__)

d2l.set_figsize()
img = Image.open('./Datasets/catdog.jpg')
w, h = img.size
print("w = %d, h = %d" % (w, h))  # w = 299, h = 231


def MultiBoxPrior(feature_map, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5]):
    '''
    Function:
        输出给定输入fmap的bounding boxes
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
    print(base_anchors)

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


X = torch.Tensor(1, 3, h, w)  # 构造输入数据
Y = MultiBoxPrior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
Y.shape  # torch.Size([1, 345345, 4])

## 上面的返回的锚框变量其形状为(1, 总的锚框个数， 4个指示的坐标位置)，
## 将返回值转换为格式（原图的高，原图的宽，以其中某一像素为中心的锚框个数，4）,
## 那么，就可以通过某个像素的坐标位置来的得到它的所有锚框了。

boxes = Y.reshape((h, w, 5, 4))
boxes[20, 200, 0, :]
# 坐标位置为(20, 200)的像素所对应的第0个锚框,返回锚框的四个坐标(左上x和y、右下x和y)
# tensor([ 0.2939, -0.2884,  1.0439,  0.4616]) 如何理解?
# 这四个坐标已经除以图像的高和宽，因此值在0到1之间
# 即0.2939和1.0439已经除了width,-0.2884和0.4616已经除了height.

## 那么这些数值又是怎样联系到sizes和ratios的呢？
# 第一个锚框的sizes[0]=0.75， retios[0]=1, 即1.0439-0.2939=0.4616+0.2884， 两者相等

boxes[20, 200, 1, :]


# tensor([ 0.1386, -0.1786,  1.1992,  0.3517])
# 第2个锚框的sizes[0]=0.75， retios[1]=2, 即1.1992-0.1386=(0.3517+0.1786)*2， 两者相等

def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    return plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0],
                         height=bbox[3]-bbox[1], fill=False, edgecolor=color,
                         linewidth=2)


def show_bboxes(axes, bboxes, labels=None, colors=None):
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

def compute_intersection(set_1, set_2):
    """
    Function:
        计算anchor之间的交集
    Args:
        set_1: a tensor of dimensions (n1, 4), anchor表示成(xmin, ymin, xmax, ymax), n1个components
        set_2: a tensor of dimensions (n2, 4), anchor表示成(xmin, ymin, xmax, ymax), n2个components
    Returns:
        intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, shape: (n1, n2)
    """

    print(set_1[:, :2].shape)  # torch.Size([n1, 2])
    print(set_2[:, :2].shape)  # torch.Size([n2, 2])
    print(set_1[:, :2].unsqueeze(1).shape)   # torch.Size([n1, 1, 2])
    print(set_2[:, :2].unsqueeze(0).shape)   # torch.Size([1, n2, 2])

    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1,n2,2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1,n2,2)
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


bbox_scale = torch.tensor((w, h, w, h), dtype=torch.float32)
ground_truth = torch.tensor([
        [0, 0.1, 0.08, 0.52, 0.92],
        [1, 0.55, 0.2, 0.9, 0.88]
    ])
anchors = torch.tensor([
        [0, 0.1, 0.2, 0.3],
        [0.15, 0.2, 0.4, 0.4],
        [0.63, 0.05, 0.88, 0.98],
        [0.66, 0.45, 0.8, 0.8],
        [0.57, 0.3, 0.92, 0.9],
    ])
fig = plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])

plt.show()


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
        为anchor分配真实的label. 这里的anchor和label信息更复杂。
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


labels = MultiBoxTarget(anchors.unsqueeze(dim=0), ground_truth.unsqueeze(dim=0))
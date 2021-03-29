"""
Created on 2/19/2021 3:18 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import math
import torch
import numpy as np
from torch import Tensor
import torch.nn as nn
import torch.nn.init as init
from util.anchors import generate_anchors

def sample(logits):
    return torch.multinomial(logits, 1)

def init_params(net):
    '''Init layer parameters.'''
    for m in net:
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0.1)

def dot_simililarity(x, y):
    v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
    # x shape: (N, 1, C)
    # y shape: (1, C, 2N)
    # v shape: (N, 2N)
    return v


def cosine_simililarity(x, y):
    # x shape: (N, 1, C)
    # y shape: (1, 2N, C)
    # v shape: (N, 2N)
    _cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
    v = _cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
    return v

def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.
    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format
    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def _sample_box(min_box_side, img_w, img_h, ar=False, min_ar=None, max_ar=None):
    width = np.random.randint(min_box_side, img_w + 1)  # [28,84]
    if not ar:
        height = width
        x = np.random.randint(0, img_w + 1 - width)  # [0, 56]
        y = np.random.randint(0, img_h + 1 - height)

    else:
        lower_h, higher_h = np.ceil(width / max_ar), np.floor(width / min_ar)
        lower_h, higher_h = max(min_box_side, lower_h), min(img_w+1, higher_h)
        height = np.random.randint(lower_h, higher_h)
        ar = width / height
        assert (ar>=min_ar and ar <=max_ar)
        x = np.random.randint(0, img_w + 1 - width)  # [0, 56]
        y = np.random.randint(0, img_h + 1 - height)

    x1, y1, x2, y2 = x, y, x + width - 1, y + height - 1
    return x1, y1, x2, y2

# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.
    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow
    Converted to Python by Aaron Snoswell
    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """
    angle = math.radians(angle)
    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )

def generate_boxes(base_size=28, min_box_side=21, feat_height=21, feat_width=21, img_size=84,
                   feat_stride=4, ratios=[0.5, 1.0, 2.0],
                   scales=np.array([1.0, 2., 2.5])):
    a = generate_anchors(base_size=base_size, ratios=ratios,
                         scales=scales)
    #print(a)
    #feat_height, feat_width = 21, 21 # 81/4
    shift_x = np.arange(0, feat_width) * feat_stride
    shift_y = np.arange(0, feat_height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                              shift_x.ravel(), shift_y.ravel())).transpose())
    shifts = shifts.contiguous().float()
    #print('shifts ', shifts)

    A = a.shape[0]
    K = shifts.size(0)

    _anchors = torch.from_numpy(a)
    # anchors = self._anchors.view(1, A, 4) + shifts.view(1, K, 4).permute(1, 0, 2).contiguous()
    anchors = _anchors.view(1, A, 4) + shifts.view(K, 1, 4)
    #print('anchors ', anchors)
    anchors = anchors.view(K * A, 4)
    anchors = torch.clip(anchors, 0, img_size-1).float()

    # filter small boxes
    keep = _filter_boxes(anchors, min_size=min_box_side)
    anchors = anchors[keep]
    return anchors

def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep

def convert_image_np(inp, norm=True):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    if norm:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
    return inp

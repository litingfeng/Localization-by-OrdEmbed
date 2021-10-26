# -*- coding: utf-8 -*-
# @Time : 9/7/21 5:13 PM
# @Author : Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.

import torch
import numpy as np
from torch import Tensor
from util.anchors import generate_anchors

def sample(logits):
    return torch.multinomial(logits, 1)

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

def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep

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

def convert_image_np(inp, norm=True):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    if norm:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def calculate_correct(diff_feat, diff_iou):
    mask = (torch.sign(diff_feat) == torch.sign(diff_iou))
    correct = mask.sum().item()
    return correct

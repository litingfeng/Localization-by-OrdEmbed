"""
Created on 3/8/2021 12:31 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import torch
import numpy as np
from util.utils import box_iou, _sample_box

class MyBoxScaleTransform:
    """
    Note: box size is random but square
    Translate and scale the box
    """
    def __init__(self, num=2, img_h=84, img_w=84, min_box_side=28):
        self.img_h, self.img_w = img_h, img_w
        self.num = num
        self.bg_thresh = 0.5
        self.bg_l_thresh = 0.1
        self.min_box_side = min_box_side
        self.min_ar, self.max_ar = 0.2, 4.0

    def __call__(self, box):
        np.random.seed()
        all_new_boxes, ious = [], []
        for i in range(self.num):
            iou_flag = np.random.randint(2)  # 0, randomly sample; 1, iou>0.5
            if iou_flag == 0:
                done = False
                while not done:
                    x1, y1, x2, y2 = _sample_box(self.min_box_side,
                                                 self.img_w, self.img_h)
                    iou = box_iou(box.view(-1, 4),
                                  torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32))
                    if iou >= self.bg_l_thresh and iou < self.bg_thresh:
                        done = True
            elif iou_flag == 1:
                done = False
                while not done:
                    x1, y1, x2, y2 = _sample_box(self.min_box_side,
                                                 self.img_w, self.img_h)
                    iou = box_iou(box.view(-1, 4),
                                  torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32))
                    if iou >= self.bg_thresh:
                        done = True
            ious.append(iou.squeeze())
            all_new_boxes.append(torch.tensor([x1, y1, x2, y2], dtype=torch.float32))
        all_new_boxes = torch.stack(all_new_boxes)
        ious = torch.stack(ious)
        return all_new_boxes, ious

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

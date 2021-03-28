"""
Created on 3/2/2021 11:05 AM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import math
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from util.utils import box_iou
from util.bbox_generator import BoxSampler
from functools import partial, update_wrapper

def test_exp():
    x = np.linspace(-1, 2, 100)
    y = np.exp(3*x)

    plt.figure()
    plt.plot(x, np.exp(3*x), label='exp(3x)')
    plt.plot(x, np.exp(x), label='exp(x)')
    plt.xlabel('$x$')
    plt.ylabel('$\exp(x)$')
    plt.show()

def _sample_box(min_box_side, img_w, img_h):
    width = np.random.randint(min_box_side, img_w + 1)  # [28,84]
    height = width
    x = np.random.randint(0, img_w + 1 - width)  # [0, 56]
    y = np.random.randint(0, img_h + 1 - height)
    x1, y1, x2, y2 = x, y, x + width - 1, y + height - 1
    return x1, y1, x2, y2

def generate_random_rois_scale(img_h, img_w, box, num, min_box_side):
    """
    generate new negative rois with random size but square
    """
    thresh = 0.5
    np.random.seed()
    all_new_boxes, ious = [], []
    for i in range(num):
        iou_flag = np.random.randint(2)  # 0, randomly sample; 1, iou>0.5
        if iou_flag == 0:
            done = False
            while not done:
                x1, y1, x2, y2 = _sample_box(min_box_side, img_w, img_h)
                iou = box_iou(box.view(-1, 4),
                              torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32))
            if iou >= 0.1 and iou < 0.5:
                done = True
        else:
            done = False
            while not done:
                x1, y1, x2, y2 = _sample_box(min_box_side, img_w, img_h)
                iou = box_iou(box.view(-1, 4),
                              torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32))
                if iou > thresh:
                    done = True
        ious.append(iou.squeeze())
        all_new_boxes.append(torch.tensor([x1, y1, x2, y2], dtype=torch.float32))
    all_new_boxes = torch.stack(all_new_boxes)
    ious = torch.stack(ious)
    return all_new_boxes, ious

def box_generator(img_h, img_w, box):
    #given a BB generate a set of BBs with desired IoUs
    Generator = BoxSampler()
    image_size = [img_w, img_h]
    B = torch.tensor([400., 200., 700., 500., 0]).cuda()
    IoUs= torch.tensor([0.5, 0.6]).cuda()
    sampled_boxes, IoUs = Generator.sample_single(box.cuda(), IoUs, image_size)
    print("sampled_boxes= ", sampled_boxes)
    print("IoUs of the sampled_boxes= ", IoUs)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_h, img_w = 84, 84
    box = torch.tensor([42, 42, 83, 83], dtype=torch.float32)
    timer1 = time.time()
    for t in range(100):
        new_box, ious = generate_random_rois_scale(img_h, img_w, box, 2, 28)
        # (num, 4), (num,)
        #break
    timer2 = time.time()
    print('time ', timer2-timer1)
    # img_h, img_w = 84, 84
    # box = torch.tensor([42, 42, 83, 83, 0], dtype=torch.float32)
    # timer1 = time.time()
    # for t in range(100):
    #     box_generator(img_h, img_w, box)
    #     # (num, 4), (num,)
    #     break
    # timer2 = time.time()
    # print('time ', timer2 - timer1)
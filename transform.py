"""
Created on 3/8/2021 12:31 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import torch
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF
from util.utils import box_iou, _sample_box
from util.bbox_generator2 import BoxSampler
from util.utils import generate_boxes

class MyRotationTransform:
    """Rotate by one of the given angles.
       rotation angle value in degrees, counter-clockwise.
    """
    def __init__(self, do_crop=False,
                 angles=np.array(range(0, 181, 15)), num=3):
        self.angles = angles
        self.do_crop = do_crop
        self.num = num
        self.default_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

    def __call__(self, x):
        np.random.seed()
        if self.num == 3:
            angle1, angle2 = tuple(np.random.choice(self.angles, size=2,
                                              replace=False))
            angle1, angle2 = int(angle1), int(angle2)
            xi = TF.rotate(x, angle1)
            xj = TF.rotate(x, angle2)

            x0 = self.default_trans(x)
            xi = self.default_trans(xi)
            xj = self.default_trans(xj)

            return angle1, angle2, x0, xi, xj
        elif self.num == 1:
            angle = int(np.random.choice(self.angles))
            xi = TF.rotate(x, angle)
            xi = self.default_trans(xi)
            return angle, xi

class MyBoxTransform:
    """
    Note: box size 28*28, image size 84*84
    Translate the box in random one direction by one of the given pixels.

    """
    def __init__(self, pixels=np.array(range(0, 181, 15)), num=2):
        self.pixels = pixels
        self.num = num

    def __call__(self, box):
        '''

        :param box: [x1,y1,x2,y2]
        :return:
        '''
        np.random.seed()
        #print('type box ', type(box))
        x1, y1, x2, y2 = box
        #print('box ', box)
        dists = np.array([x1, y1, 83-x2, 83-y2])
        #print('dists ', dists)
        # random sample from left/up/right/down
        direction = np.random.randint(4)
        #print('dire ', direction)
        dist = dists[direction]
        if self.num == 2:
            if dist > 1:
                pix1, pix2 = tuple(np.random.choice(range(dist), size=2,
                                                    replace=False))
            else:
                pix1 = pix2 = dist
            if direction == 0: # left
                new_box1, new_box2 = np.array([x1-pix1, y1, x2-pix1, y2]), \
                                     np.array([x1-pix2, y1, x2-pix2, y2])
            elif direction == 1: # up
                new_box1, new_box2 = np.array([x1, y1-pix1, x2, y2-pix1]), \
                                     np.array([x1, y1-pix2, x2, y2-pix2])
            elif direction == 2: # right
                new_box1, new_box2 = np.array([x1+pix1, y1, x2+pix1, y2]), \
                                     np.array([x1+pix2, y1, x2+pix2, y2])
            elif direction == 3: # down
                new_box1, new_box2 = np.array([x1, y1+pix1, x2, y2+pix1]), \
                                     np.array([x1, y1+pix2, x2, y2+pix2])
            # print('new_box1, new_box2 ', new_box1, new_box2)
            # exit()
            return new_box1, new_box2, pix1, pix2
        elif self.num == 1:
            if dist > 1:
                pix1 = np.random.choice(range(dist),replace=False)
            else:
                pix1 = dist
            if direction == 0: # left
                new_box1 = np.array([x1-pix1, y1, x2-pix1, y2])

            elif direction == 1: # up
                new_box1 = np.array([x1, y1-pix1, x2, y2-pix1])

            elif direction == 2: # right
                new_box1 = np.array([x1+pix1, y1, x2+pix1, y2])

            elif direction == 3: # down
                new_box1 = np.array([x1, y1+pix1, x2, y2+pix1])

            return new_box1, pix1

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
                                                 self.img_w, self.img_h,
                                                 ar=True, min_ar=self.min_ar,
                                                 max_ar=self.max_ar)
                    # x1, y1, x2, y2 = _sample_box(self.min_box_side,
                    #                              self.img_w, self.img_h)
                    iou = box_iou(box.view(-1, 4),
                                  torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32))
                    if iou >= self.bg_l_thresh and iou < self.bg_thresh:
                        done = True
            elif iou_flag == 1:
                done = False
                while not done:
                    x1, y1, x2, y2 = _sample_box(self.min_box_side,
                                                 self.img_w, self.img_h,
                                                 ar = True, min_ar = self.min_ar,
                                                 max_ar = self.max_ar)
                    # x1, y1, x2, y2 = _sample_box(self.min_box_side,
                    #                              self.img_w, self.img_h)
                    iou = box_iou(box.view(-1, 4),
                                  torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32))
                    if iou >= self.bg_thresh:
                        done = True
            ious.append(iou.squeeze())
            all_new_boxes.append(torch.tensor([x1, y1, x2, y2], dtype=torch.float32))
        all_new_boxes = torch.stack(all_new_boxes)
        ious = torch.stack(ious)
        return all_new_boxes, ious

# class MyBoxScaleARTransform:
#     """
#     Note: box size and aspect ratio is random
#     Translate and scale the box
#     """
#     def __init__(self, , num=2):
#         self.num = num
#         self._anchors = anchors
#
#     def __call__(self, box):
#         np.random.seed()
#         all_new_boxes, ious = [], []

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

if __name__ == "__main__":
    import torch
    import time
    target_transform = MyBoxTransform()
    box = torch.tensor([14,14,42,42], dtype=torch.int16)
    timer1 = time.time()
    for t in range(100):
        target_new = target_transform(box)
    timer2 = time.time()
    print('time ', timer2 - timer1)
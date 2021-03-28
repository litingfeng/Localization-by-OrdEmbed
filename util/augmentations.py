"""
Created on 10/7/2020 8:16 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
from PIL import Image
import numpy as np
import cv2
import random
import torchvision.transforms.functional as TF
from util.data_aug import *
from util.bbox_util import *

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        target[2] = target[0] + target[2] - 1
        target[3] = target[1] + target[3] - 1
        target = target.reshape(1,4)
        # convert pil to cv2
        img = np.array(img)
        img = img[:, :, ::-1].copy() if len(img.shape) == 3 else img.copy()
        for t in self.transforms:
            img, target = t(img, target)
        # convert back
        img = Image.fromarray(img[:, :, ::-1]) if len(img.shape) == 3 else Image.fromarray(img)
        return img, target[0]

class resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, target):
        img = TF.resize(img, self.size, self.interpolation)
        return img


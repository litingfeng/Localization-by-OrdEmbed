"""
Created on 5/11/2021 9:59 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import os
import numpy as np
from torchvision import transforms
import json
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import transforms
from PIL import Image, ImageDraw
from util.utils import convert_image_np
from matplotlib.patches import Rectangle
from util.augmentations import Compose, resize
from util.utils import box_iou
import torch
from datasets.batch_sampler import PrototypicalBatchSampler

DEBUG = False

class CUB_fewshot():
    def __init__(self, root, img_size, anchors=None, loose=0,
                 mode='base', transform=None):
        self.transform = transform
        self.img_size = img_size
        self.loose = loose

        boxes = np.loadtxt(os.path.join(root, 'bounding_boxes.txt'))
        images = pd.read_csv(os.path.join(root, 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])

        images.set_index("filepath", inplace=True)
        if mode == 'base':
            data_file = os.path.join(root, 'fewshot', 'warbler_train15.json')
        else:
            data_file = os.path.join(root, 'fewshot', 'warbler_val5.json'.format(mode))
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        print("{} images {} labels {}".format(mode, len(self.meta['image_names']),
              len(set(self.meta['image_labels']))))


        self.data, self.target = [], []
        for i in range(len(self.meta['image_names'])):
            image_path = os.path.join(self.meta['image_names'][i])
            img = Image.open(image_path)
            if img.getbands()[0] == 'L':
                img = img.convert('RGB')
            self.data.append(img.copy())
            img.close()

            this_id = images.loc[image_path.split('images/')[1]]
            self.target.append(boxes[this_id-1, 1:].flatten())

        self.target = np.vstack(self.target)
        # convert target[x, y, w, h]->[x1, y1, x2, y2]
        self.target[:, 2] = self.target[:, 0] + self.target[:, 2] - 1
        self.target[:, 3] = self.target[:, 1] + self.target[:, 3] - 1

        # deal with class label, map to start from 0
        self.labels = []
        self.cls2idx = {cls: i for i, cls in enumerate(set(self.meta['image_labels']))}

        self.anchors = anchors
        ''' for each image, compute iou(gt, all anchors)'''
        target = torch.from_numpy(self.target.copy())
        self.target_org = self.target.copy()
        # convert target to 224 images,
        for i in range(target.shape[0]):
            img = self.data[i]
            width, height = img.size
            target[i, 0] /= width
            target[i, 2] /= width
            target[i, 1] /= height
            target[i, 3] /= height
        target = target * self.img_size
        self.target_org = target.clone()
        if self.loose == 1:
            # enlarge box
            target[:, 0] -= 10
            target[:, 1] -= 10
            target[:, 2] += 10
            target[:, 3] += 10
            target = torch.clip(target, 0, self.img_size-1)

        self.ious = box_iou(target, anchors)

        self.iou_map = [{i: None for i in range(10)} for _ in range(target.shape[0])]
        for i in range(target.shape[0]):
            for j in range(10):
                inds = torch.where((self.ious[i] >= j * 0.1) & (self.ious[i] < (j + 1) * 0.1))[0]
                if len(inds) != 0:
                    self.iou_map[i][j] = inds
                else:
                    del self.iou_map[i][j]

        # if DEBUG:
        #     print('num anchors ', anchors.shape)
        #     print('targets ', target[0], self.target[0])
        #     print('self.iou_map[0][3] ', self.iou_map[0][3])
        #     print(self.ious[0, self.iou_map[0][3]])
        #     for j in range(10):
        #         if j in self.iou_map[0].keys():
        #             print(len(self.iou_map[0][j]))
        #         else:
        #             print(j)
        #     exit()

    def __getitem__(self, index):
        img, target, label = self.data[index], self.target[index].copy(), \
                             self.meta['image_labels'][index]

        iou_map, ious = self.iou_map[index], self.ious[index]
        if self.transform is not None:
            img, target, _ = self.transform(img, np.expand_dims(target, axis=0))

        target = target * self.img_size
        if self.loose == 1:
            target [0, 0] -= 10
            target[0, 1] -= 10
            target[0, 2] += 10
            target[0, 3] += 10
            target = np.clip(target, 0, self.img_size-1)

        if DEBUG:
            print('after img ', img.shape)
            print(img.min(), img.max())
            img_ = convert_image_np(torchvision.utils.make_grid(img.unsqueeze(0)),
                                    norm=True)
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(img_)
            x1, y1, x2, y2 = target[0]
            patch = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=5,
                              edgecolor='g', facecolor='none', fill=False)
            ax.add_patch(patch)
            x1, y1, x2, y2 = self.target_org[index]
            patch = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=5,linestyle='--',
                              edgecolor='g', facecolor='none', fill=False)
            ax.add_patch(patch)

        # random sample two ious and two box with these ious
        keys = list(iou_map.keys())
        smp_iou = np.random.choice(keys, size=2)
        # print('smp iou ', smp_iou)
        ind1, ind2 = np.random.choice(iou_map[smp_iou[0]]), \
                     np.random.choice(iou_map[smp_iou[1]])
        # print('inds12 ', ind1, ind2)
        ti, tj, iou1, iou2 = self.anchors[ind1], self.anchors[ind2], \
                             ious[ind1], ious[ind2]
        if DEBUG:
            print('ti ', ti, ' tj ', tj)
            print('ious ', iou1, ' ', iou2)
            patch = Rectangle((ti[0], ti[1]), ti[2]-ti[0], ti[3]-ti[1], linewidth=1,
                              edgecolor='g', facecolor='none', fill=False)
            #ax.add_patch(patch)
            patch = Rectangle((tj[0], tj[1]), tj[2] - tj[0], tj[3] - tj[1], linewidth=1,
                              edgecolor='y', facecolor='none', fill=False)
            #ax.add_patch(patch)
            plt.axis('off')
            plt.savefig('ann_train.png', bbox_inches='tight', dpi=400)
            plt.show()
            exit()

        return img, np.hstack((np.squeeze(target),
                        np.array([self.cls2idx[label]]))), \
               ti, tj, iou1, iou2
        # return img, np.hstack((np.squeeze(target) * self.img_size,
        #                        np.array([self.cls2idx[label]]))), \
        #        np.stack((ti, tj)), np.array([iou1, iou2])

    def __len__(self):
        return len(self.data)

def init_sampler(classes_per_set, samples_per_class, labels, iterations, mode):
    if 'train' in mode:
        classes_per_it = classes_per_set
        num_samples = samples_per_class
    else:
        classes_per_it = classes_per_set
        num_samples = samples_per_class

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=iterations)

if __name__ =='__main__':
    from util.utils import generate_boxes
    from util.data_aug import *
    from util.bbox_util import *
    from util import joint_transforms as t
    from constants import IMAGENET_MEAN, IMAGENET_STD

    anchors = generate_boxes(base_size=16, feat_height=14, feat_width=14,
                             min_box_side=25, img_size=224,
                             feat_stride=16, ratios=np.linspace(0.3, 3.5, num=15),
                             scales=np.array(range(2, 13)))
    # anchors = generate_boxes(base_size=4,
    #                          feat_height=21, feat_width=21, img_size=84,
    #                          feat_stride=4,
    #                          ratios=np.linspace(0.3, 3.5, num=15),
    #                          min_box_side=16,
    #                          scales=np.array(range(4, 20)))
    dataset = CUB_fewshot('/research/cbim/vast/tl601/Dataset/CUB_200_2011',
                          img_size=84,mode='base', anchors=anchors, loose=1,
                          transform=t.Compose([
                              t.ConvertFromPIL(),
                              t.ToPercentCoords(),
                              t.Resize(84),
                              t.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                              t.ToTensor()  # no change to (0, 1)
                          ])
                          )
    train_batchsampler = init_sampler(5, 5, dataset.meta['image_labels'], 100, 'train')
    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, **kwargs)
        #batch_sampler=train_batchsampler)


    iouis, ioujs = [], []
    ars = []
    ws, hs = [], []
    for i, (data, target,ti, tj, ioui, iouj) in enumerate(train_loader):
        print('data ', data.shape, ' target ', target)
        exit()



# -*- coding: utf-8 -*-
# @Time : 9/11/21 11:31 AM
# @Author : Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
import os, json
import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from util.utils import convert_image_np
from matplotlib.patches import Rectangle
from util.utils import box_iou
from datasets.batch_sampler import PrototypicalBatchSampler

DEBUG = False
class CUB():
    def __init__(self, root, img_size, bg_name=None, anchors=None, support_size=5,
                  mode='base',transform=None):
        self.transform = transform
        self.img_size  = img_size

        boxes  = np.loadtxt(os.path.join(root, 'bounding_boxes.txt')) # load gt boxes
        images = pd.read_csv(os.path.join(root, 'images.txt'), sep=' ', names=['img_id', 'filepath'])
        images.set_index('filepath', inplace=True)
        if mode == 'base': # pretrain or train agent
            data_file = os.path.join(root, 'fewshot', 'warbler_train15.json')
        else:
            if bg_name == 'warbler':
                datalist = 'warbler_val5.json'
            elif bg_name == 'gull':
                datalist = 'gull_59_64.json'
            elif bg_name == 'wren':
                datalist = 'wren_193_197.json'
            elif bg_name == 'sparrow':
                datalist = 'sparrow_114_119.json'
            elif bg_name == 'oriole':
                datalist  = 'oriole_95_98.json'
            elif bg_name == 'kingfisher':
                datalist = 'kingfisher_79_83.json'
            elif bg_name == 'vireo':
                datalist = 'vireo_151_157.json'
            else:
                print('background {} not implemented'.format(bg_name))
                exit()
            print('loading {}...'.format(datalist))
            data_file = os.path.join(root, 'fewshot', datalist)
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

        self.cls2idx = {cls: i for i, cls in enumerate(set(self.meta['image_labels']))}

        target = torch.from_numpy(self.target.copy())
        self.target_org = self.target.copy()
        # convert target to img_size images,
        for i in range(target.shape[0]):
            img = self.data[i]
            width, height = img.size
            target[i, 0] /= width
            target[i, 2] /= width
            target[i, 1] /= height
            target[i, 3] /= height
        target = target * self.img_size
        self.target_org = target.clone()


        self.anchors = None
        if anchors is not None: # pretrain
            self.anchors = anchors
            self.ious = box_iou(target, anchors)
            self.iou_map = [{i: None for i in range(10)} for _ in range(target.shape[0])]
            for i in range(target.shape[0]):
                for j in range(10):
                    inds = torch.where((self.ious[i] >= j * 0.1) & (self.ious[i] < (j + 1) * 0.1))[0]
                    if len(inds) != 0:
                        self.iou_map[i][j] = inds
                    else:
                        del self.iou_map[i][j]
        else: # adaptation
            # sample support data
            inds = np.random.permutation(len(self.data))[:support_size]
            self.supp_data_array = [self.data[i].copy() for i in inds]
            self.supp_targets_ = self.target[inds[:support_size]].copy()

            # convert data to model input
            self.supp_data, self.supp_targets = [], np.zeros((support_size, 4))
            for i in range(len(self.supp_data_array)):
                img, target = self.supp_data_array[i], self.supp_targets_[i]
                if self.transform is not None:
                    img, target, _ = self.transform(img, np.expand_dims(target, axis=0))
                target = (np.squeeze(target)) * self.img_size
                self.supp_data.append(img)
                self.supp_targets[i] = target
            self.supp_data = torch.stack(self.supp_data)
            self.supp_targets = torch.from_numpy(self.supp_targets)

    def __getitem__(self, index):
        img, target, label = self.data[index], self.target[index].copy(), \
                             self.meta['image_labels'][index]
        if self.anchors is not None: # pretrain
            iou_map, ious = self.iou_map[index], self.ious[index]
        if self.transform is not None:
            img, target, _ = self.transform(img, np.expand_dims(target, axis=0))
            target = target * self.img_size

        if DEBUG:
            print('transformed image size: ', img.shape)
            print('target ', target)
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
                              edgecolor='b', facecolor='none', fill=False)
            ax.add_patch(patch)

        if self.anchors is not None: # pretrain
            '''random sample two ious and two box with these ious'''
            keys = list(iou_map.keys())
            smp_iou = np.random.choice(keys, size=2)
            # print('smp iou ', smp_iou)
            ind1, ind2 = np.random.choice(iou_map[smp_iou[0]]), \
                         np.random.choice(iou_map[smp_iou[1]])
            # print('inds12 ', ind1, ind2)
            ti, tj, ioui, iouj = self.anchors[ind1], self.anchors[ind2], \
                                 ious[ind1], ious[ind2]
            if DEBUG:
                print('ti ', ti, ' tj ', tj)
                print('ious ', ioui, ' ', iouj)
                patch = Rectangle((ti[0], ti[1]), ti[2]-ti[0], ti[3]-ti[1], linewidth=1,
                                  edgecolor='g', facecolor='none', fill=False)
                ax.add_patch(patch)
                patch = Rectangle((tj[0], tj[1]), tj[2] - tj[0], tj[3] - tj[1], linewidth=1,
                                  edgecolor='r', facecolor='none', fill=False)
                ax.add_patch(patch)
                plt.axis('off')
                plt.savefig('ann_train.png', bbox_inches='tight', dpi=400)
                plt.show()
                exit()

            return img, np.hstack((np.squeeze(target),
                                   np.array([self.cls2idx[label]]))), \
                   ti, tj, ioui, iouj
        else:
            return img, np.hstack((np.squeeze(target), np.array([label])))

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

    anchors = generate_boxes(base_size=16, feat_height=14, feat_width=14,
                             min_box_side=25, img_size=224,
                             feat_stride=16, ratios=np.linspace(0.3, 3.5, num=15),
                             scales=np.array(range(2, 13)))
    transform = t.Compose([
        t.ConvertFromPIL(),
        t.ToPercentCoords(),
        t.Resize(224),
        t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        t.ToTensor()  # no change to (0, 1)
    ])
    trainset = CUB('/research/cbim/vast/tl601/Dataset/CUB_200_2011',
                   bg_name='warbler', anchors=anchors, mode='base', img_size=224,
                   transform=transform)

    train_batchsampler = init_sampler(5, 5, trainset.meta['image_labels'], 100, 'train')
    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=2, shuffle=True, **kwargs)
    #batch_sampler=train_batchsampler)


    iouis, ioujs = [], []
    ars = []
    ws, hs = [], []
    for i, (data, target,ti, tj, ioui, iouj) in enumerate(train_loader):
        print('data ', data.shape, ' target ', target)
        exit()


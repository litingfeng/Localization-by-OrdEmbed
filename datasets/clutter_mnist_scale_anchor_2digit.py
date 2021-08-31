"""
Created on 3/11/2021 11:56 AM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import os
import torch
from PIL import Image
import random
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from util.utils import box_iou

DEBUG = False

class MNIST_CoLoc(datasets.MNIST):

    def __init__(self, root, train=True, anchors=None, bg_name=None,
                 sample_size='whole',
                 transform=None, target_transform=None,
                 download=False, digit=4, clutter=0, datapath=None):
        super(MNIST_CoLoc, self).__init__(root, train=train, download=download,
                                          transform=transform, target_transform=target_transform)
        # make sure calling of transform, after which target [x,y,w,h]->[x1,y1,x2,y2]
        assert (self.transform is not None)

        self.digit = digit
        self.datapath = datapath
        self.totensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        phase = 'train' if train else 'test'
        datadir = os.path.join(self.datapath, phase)
        if clutter == 1:
            # data with fixed size digit 28*28
            print('loading {} data... '.format(bg_name))
            if 'patch' not in bg_name:
                self.new_digit_data = np.load(os.path.join(datadir, bg_name,
                                                           str(digit) + '_{}_data.npy'.format(bg_name))) #
                self.new_targets_all = np.load(os.path.join(datadir, bg_name,
                                                        str(digit) + '_{}_label.npy'.format(bg_name))) #'_label.npy'))
            else:
                self.new_digit_data = np.load(os.path.join(datadir, bg_name,
                                                           str(digit) + '_data.npy'))
                self.new_targets_all = np.load(os.path.join(datadir, bg_name,
                                                        str(digit) + '_label.npy'))

        self.new_targets_all = torch.from_numpy(self.new_targets_all)
        if '2digit' in bg_name:
            self.new_targets = self.new_targets_all[self.new_targets_all[:, :, -1] == self.digit].reshape(-1,5)
            self.new_targets_noise = self.new_targets_all[self.new_targets_all[:, :, -1] != self.digit].reshape(-1, 5)

        if train and sample_size != 'whole':
            # inds = np.random.permutation(len(self.new_digit_data))
            # np.save(os.path.join(datadir, 'moreclutter', str(digit) + '_inds'), arr=inds)
            inds = np.load(os.path.join(datadir, 'moreclutter', str(digit) + '_inds.npy'))
            sample_size = int(sample_size)
            self.new_digit_data = self.new_digit_data[inds[:sample_size]]
            self.new_targets = self.new_targets[inds[:sample_size]]
            self.new_targets_noise = self.new_targets_noise[inds[:sample_size]]

        self.anchors = anchors
        # for each image, compute iou(gt, all anchors)
        new_targets = self.new_targets[:, :4].clone()
        new_targets[:,2] = new_targets[:, 2] + new_targets[:, 0] - 1
        new_targets[:, 3] = new_targets[:, 3] + new_targets[:, 1] - 1
        self.ious = box_iou(new_targets, anchors)
        # for each image, sort according to iou, group then save in dict
        self.iou_map = [{i: None for i in range(10)} for _ in range(new_targets.shape[0])]
        for i in range(new_targets.shape[0]):
            for j in range(10):
                inds = torch.where((self.ious[i]>=j*0.1) & (self.ious[i] < (j+1)*0.1 ))[0]
                if len(inds) != 0 :
                    self.iou_map[i][j] = inds
                else:
                    del self.iou_map[i][j]
        # if DEBUG:
        #     print('num anchors ', anchors.shape)
        #     print('targets ', new_targets[0], self.new_targets[0])
        #     print('self.iou_map[0][3] ', self.iou_map[0][3])
        #     print(self.ious[0, self.iou_map[0][3]])
        #     for j in range(10):
        #         if j in self.iou_map[0].keys():
        #             print(len(self.iou_map[0][j]))
        #         else:
        #             print(j)
        #     exit()

    def __getitem__(self, index):
        img, target, target_noise = self.new_digit_data[index], self.new_targets[index].numpy(), \
                                    self.new_targets_noise[index].numpy()
        #target = np.stack((target_noise, target))
        target = np.stack((target, target_noise))
        iou_map, ious = self.iou_map[index], self.ious[index]
        img = Image.fromarray(img)#.convert(mode='RGB')
        #print('org target ', target)
        if self.transform is not None:
            img, target = self.transform(img, target)

        if DEBUG:
            f, ax = plt.subplots()
            x1, y1, x2, y2, _ = target[0]
            print('new target ', target)
            ax.imshow(np.asarray(img), cmap='gray', interpolation='none')
            patch = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,
                              edgecolor='g', facecolor='none', fill=False)
            ax.add_patch(patch)
            x1, y1, x2, y2, _ = target[1]
            patch = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,
                              edgecolor='g', facecolor='none', fill=False)
            ax.add_patch(patch)

        img = self.totensor(img)

        # random sample two ious and two box with these ious
        keys = list(iou_map.keys())
        smp_iou = np.random.choice(keys, size=2)
        ind1, ind2 = np.random.choice(iou_map[smp_iou[0]]), \
                     np.random.choice(iou_map[smp_iou[1]])
        ti, tj, iou1, iou2 = self.anchors[ind1], self.anchors[ind2], \
                             ious[ind1], ious[ind2]

        if DEBUG:
            print('ti ', ti, ' tj ', tj)
            print('ious ', iou1, ' ', iou2)
            patch = Rectangle((ti[0], ti[1]), ti[2]-ti[0], ti[3]-ti[1], linewidth=1,
                              edgecolor='r', facecolor='none', fill=False)
            ax.add_patch(patch)
            patch = Rectangle((tj[0], tj[1]), tj[2] - tj[0], tj[3] - tj[1], linewidth=1,
                              edgecolor='y', facecolor='none', fill=False)
            ax.add_patch(patch)
            plt.show()
            exit()
        #print('target new ', target)
        return img, target, np.stack((ti, tj)), np.stack((iou1, iou2))

    def __len__(self):
        return self.new_digit_data.shape[0]

    @property
    def raw_folder(self):
        return os.path.join('/research/cbim/vast/tl601/Dataset', 'MNIST', 'raw')

    @property
    def processed_folder(self):
        return os.path.join('/research/cbim/vast/tl601/Dataset', 'MNIST', 'processed')


if __name__ == '__main__':
    import torch
    from util.data_aug import *
    from util.augmentations import Compose
    from transform import MyBoxScaleTransform
    from util.utils import generate_boxes
    kwargs = {'num_workers': 0, 'pin_memory': True}

    train_transform = Compose([Resize(84),
                               #RandomHorizontalFlip(),
                               ])
    test_transform = Compose([Resize(84)])
    anchors = generate_boxes(base_size=4,
                             feat_height=21, feat_width=21, img_size=84,
                             feat_stride=4,
                             ratios=[1.0],
                             min_box_side=28,
                             scales=np.array(range(7, 20)))

    print('number of anchors for 84*84 image ', anchors.shape[0])
    print('anchors ', anchors, torch.min(anchors[:,2:]))
    #exit()
    trainset = MNIST_CoLoc(root='.', train=True, digit=3, anchors=anchors,
                           sample_size='whole', bg_name='random_patch_2digit',
                           datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                           clutter=1, transform=train_transform
                           )
    #print('min_box_side ', torch.min(trainset.new_targets[:, 2:])) # 21
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=True, **kwargs)

    iouis, ioujs = [], []
    ars = []
    for i, (data, target, boxes, ious ) in enumerate(train_loader):
        print('target ', target.shape, ' boxes ', boxes.shape, ' ious ', ious.shape)
        exit()
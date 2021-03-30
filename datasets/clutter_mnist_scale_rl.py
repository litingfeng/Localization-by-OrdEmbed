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

DEBUG = False

class MNIST_CoLoc(datasets.MNIST):

    def __init__(self, root, train=True, transform=None, target_transform=None,
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
        self.crop = transforms.RandomResizedCrop((84, 84), scale=(0.1, 1.0))
        assert (self.datapath is not None)

        phase = 'train' if train else 'test'
        datadir = os.path.join(self.datapath, phase)
        if clutter == 1:
            # # data with fixed size digit 28*28
            # self.new_digit_data = np.load(os.path.join(datadir, 'moreclutter',
            #                                            str(digit) + '_moreclutter_data.npy'))
            # self.new_targets = np.load(os.path.join(datadir, 'moreclutter',
            #                                         str(digit) + '_moreclutter_label.npy'))
            # different scale and aspect ratio
            self.new_digit_data = np.load(os.path.join(datadir, 'clutterScaleARLarger',
                                                       str(digit) + '_clutterScaleARLarger_data.npy'))
            self.new_targets = np.load(os.path.join(datadir, 'clutterScaleARLarger',
                                                    str(digit) + '_clutterScaleARLarger_label.npy'))
        self.new_targets = torch.from_numpy(self.new_targets)

    def __getitem__(self, index):
        img, target = self.new_digit_data[index], self.new_targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img, target = self.transform(img, target)

        if DEBUG:
            f, ax = plt.subplots()
            x1, y1, x2, y2 = target
            ax.imshow(np.asarray(img), cmap='gray', interpolation='none')
            patch = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,
                              edgecolor='g', facecolor='none', fill=False)
            ax.add_patch(patch)

        img = self.totensor(img)

        if self.target_transform is not None:
            target_new = self.target_transform(target)
            ti, iou1 = target_new[0][0], target_new[1][0]
            return img, target, ti, iou1

        return img, target

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
    from util.transform import MyBoxScaleTransform
    kwargs = {'num_workers': 8, 'pin_memory': True}

    train_transform = Compose([Resize(84)])
    test_transform = Compose([Resize(84)])
    trainset = MNIST_CoLoc(root='.', train=True, digit=4,
                           datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                           clutter=1, transform=train_transform,
                           target_transform=MyBoxScaleTransform(num=1))
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=12, shuffle=True, **kwargs)

    for i, (data, target, ti, iou1 ) in enumerate(train_loader):
        print('iou1 ', iou1.shape)
        print('ti ', ti.shape)
        break




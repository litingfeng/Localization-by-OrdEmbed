"""
few shot setting for train on new digits
Created on 4/27/2021 11:21 AM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import os, torch
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

DEBUG = False

class MNIST_Supp(datasets.MNIST):
    def __init__(self, root, datapath, digit, anchors=None, batch_size=100,
                 samples_per_class=20,
                 batch_num=1000, train=True, transform=None, target_transform=None,
                 download=False, use_cache=True):
        super(MNIST_Supp, self).__init__(root, train=train, download=download,
                                         transform=transform,
                                         target_transform=target_transform)
        # make sure calling of custom transform, after which target [x,y,w,h]->[x1,y1,x2,y2]
        assert (self.transform is not None)
        self.anchors = anchors
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.samples_per_class = samples_per_class
        self.totensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # data with fixed size digit 28*28
        phase = 'train' if train else 'test'
        datadir = os.path.join(datapath, phase)
        self.digit_data = np.load(os.path.join(datadir, 'moreclutter',
                                                   str(digit) + '_moreclutter_data.npy'))
        self.digit_targets = np.load(os.path.join(datadir, 'moreclutter',
                                                str(digit) + '_moreclutter_label.npy'))
        # different scale and aspect ratio
        # self.new_digit_data = np.load(os.path.join(datadir, 'clutterScaleARLarger',
        #                                            str(digit) + '_clutterScaleARLarger_data.npy'))
        # self.new_targets = np.load(os.path.join(datadir, 'clutterScaleARLarger',
        #                                         str(digit) + '_clutterScaleARLarger_label.npy'))
        self._normalization() # data (num_img, 1, 84, 84), targets (num_img, 4)

        self.index = 0
        self.dataset_cache = self._load_data_cache()


    def _normalization(self):
        # normalize data and perform target transform for model input
        self.new_digit_data = []
        for i in range(self.digit_data.shape[0]):
            img, target = self.digit_data[i], self.digit_targets[i]
            img = Image.fromarray(img)
            if self.transform is not None:
                img, target = self.transform(img, target)
            img = self.totensor(img)
            self.new_digit_data.append(img)
        self.new_digit_data = torch.stack(self.new_digit_data)  # (num_img, 1, 84, 84)
        self.digit_targets = torch.from_numpy(self.digit_targets)

    def _load_data_cache(self):
        '''
        Collects batch_num  data for learning
        :return: A list with [support_set_x, support_set_y, target_x, target_y]
        '''
        data_cache = []
        samples_idx = np.arange(self.new_digit_data.shape[0])
        for _ in range(self.batch_num):
            support_set_x = torch.zeros(self.batch_size, self.samples_per_class, 1, 84, 84)
            support_set_y = torch.zeros(self.batch_size, self.samples_per_class, 4)
            target_x = torch.zeros(self.batch_size, 1, 84, 84)
            target_y = torch.zeros(self.batch_size, 4)
            for i in range(self.batch_size):
                choose_samples = np.random.choice(samples_idx, size=self.samples_per_class+1,
                                                  replace=False)
                x_temp = self.new_digit_data[choose_samples]
                y_temp = self.digit_targets[choose_samples]
                support_set_x[i] = x_temp[:-1]
                support_set_y[i] = y_temp[:-1]
                target_x[i]      = x_temp[-1]
                target_y[i]      = y_temp[-1]
            data_cache.append([support_set_x, support_set_y, target_x, target_y])

        return data_cache

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

    kwargs = {'num_workers': 0, 'pin_memory': True}

    train_transform = Compose([Resize(84)])
    test_transform = Compose([Resize(84)])
    trainset = MNIST_Supp('.',
                datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                 digit=4, batch_size=100, samples_per_class=20,
                 batch_num=1000, train=True, transform=train_transform,
                 use_cache=True)

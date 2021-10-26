# -*- coding: utf-8 -*-
# @Time : 9/7/21 3:31 PM
# @Author : Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.

import os, torch
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from util.data_aug import *
from util.augmentations import Compose
from util.utils import box_iou
from util.corruptions import impulse_noise, fog, gaussian_noise
from matplotlib.patches import Rectangle


DEBUG = True
IMG_SIZE = 84
DIGIT_SIZE = 28

def round_and_astype(x):
    return np.round(x).astype(np.uint8)

class MNIST_Corrupted(datasets.MNIST):

    def __init__(self, root, train, digit, bg_name, transform=None, target_transform=None,
                 download=False):
        super(MNIST_Corrupted, self).__init__(root, train=train, download=download,
                                              transform=transform, target_transform=target_transform)
        # make sure calling of transform, after which target [x,y,w,h]->[x1,y1,x2,y2]
        assert (self.transform is not None)

        self.digit = digit
        self.totensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        phase = 'train' if train else 'test'
        data_dir = os.path.join(savepath, phase)

        self.digit_data = [idx for idx, digit in enumerate(self.targets)
                           if digit == self.digit]
        print('total number of digit-{} image\t{}'.format(self.digit, len(self.digit_data)))

        self.new_digit_data = np.load(os.path.join(data_dir, bg_name, str(digit) + '_data.npy'))
        self.new_targets = np.load(os.path.join(data_dir, bg_name, str(digit) + '_label.npy'))


    def __getitem__(self, index):
        img, img_org, target = self.new_digit_data[index], self.data[self.digit_data[index]].numpy(), \
                               self.new_targets[index].reshape(1, 4)
        print('target ', target)
        print('self.data[self.digit_data[index]] ', self.data[self.digit_data[index]].shape)
        img, img_org = Image.fromarray(img), Image.fromarray(img_org)

        if self.transform is not None:
            img, target = self.transform(img, target)

        if DEBUG:

            x1, y1, x2, y2 = target[0]
            print('box ', x1, y1, x2, y2)
            f, ax = plt.subplots()
            ax.imshow(np.asarray(img), cmap='gray', interpolation='none')
            patch = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,
                              edgecolor='r', facecolor='none', fill=False)
            ax.add_patch(patch)
            patch = Rectangle((x1+5, y1+5), x2 - x1-10, y2 - y1-10, linewidth=2,
                              edgecolor='y', facecolor='none', fill=False)
            ax.add_patch(patch)
            plt.title('red')
            plt.show()
            f, ax = plt.subplots()
            ax.imshow(np.asarray(img_org), cmap='gray', interpolation='none')
            plt.show()

        img = self.totensor(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="mnist",
                        choices=['mnist']) #TODO: 'omg', 'cub'
    parser.add_argument('--bg_name', type=str, default="impulse_noise",
                        choices=['clean','clutter', 'patch',
                                 'gaussian_noise', 'impulse_noise'],
                        help='type of background of dataset')
    args = parser.parse_args()

    # generate synthesized images and save
    savepath = '/research/cbim/vast/tl601/Dataset/Synthesis_mnist_github'
    os.makedirs(os.path.join(savepath, 'train', args.bg_name), exist_ok=True)
    os.makedirs(os.path.join(savepath, 'test', args.bg_name), exist_ok=True)

    dataset = MNIST_Corrupted(root='/research/cbim/vast/tl601/Dataset',
                                          bg_name=args.bg_name, train=True,
                                          digit=0, transform=Compose([Resize(IMG_SIZE)]))
    kwargs = {'num_workers': 0, 'pin_memory': True}
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, **kwargs)

    for i, (data, target) in enumerate(data_loader):
        break

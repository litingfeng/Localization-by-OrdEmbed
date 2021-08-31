"""
Created on 5/23/2021 7:23 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import os, torch
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

DEBUG = False
class MNIST_CoLoc(datasets.MNIST):
    def __init__(self, root, digit1, digit2, train=True, transform=None,
                 target_transform=None, sample_size='whole',
                 download=False):
        super(MNIST_CoLoc, self).__init__(root, train=train, download=download,
                                          transform=transform,
                                          target_transform=target_transform)

        self.digit_data = self.data[torch.where(self.targets == digit1)]
        print('total number of digit-{} image\t{}'.format(digit1, len(self.digit_data)))
        self.digit_data2 = self.data[torch.where(self.targets == digit2)]
        print('total number of digit-{} image\t{}'.format(digit2, len(self.digit_data2)))

        if sample_size != 'whole':
            sample_size = int(sample_size)
            inds = np.random.permutation(len(self.digit_data))
            self.digit_data = self.digit_data[inds[:sample_size]]
            inds = np.random.permutation(len(self.digit_data2))
            self.digit_data2 = self.digit_data2[inds[:sample_size]]

        self.data = torch.cat((self.digit_data, self.digit_data2))
        self.targets = torch.cat((torch.zeros(self.digit_data.shape[0]),
                                  torch.ones(self.digit_data2.shape[0])))


    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join('/research/cbim/vast/tl601/Dataset', 'MNIST', 'raw')

    @property
    def processed_folder(self):
        return os.path.join('/research/cbim/vast/tl601/Dataset', 'MNIST', 'processed')

if __name__ == '__main__':
    import torch
    from util.data_aug import *
    from datasets.batch_sampler import BalancedBatchSampler

    kwargs = {'num_workers': 0, 'pin_memory': True}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = MNIST_CoLoc(root='.', train=False, digit1=3, digit2=0, transform=transform)

    train_batch_sampler = BalancedBatchSampler(trainset.targets, n_classes=2, n_samples=5)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_sampler=train_batch_sampler, **kwargs)

    for i, (data, target) in enumerate(train_loader):
        print(type(target))
        print('data ', data.shape, 'target ', target)
        break


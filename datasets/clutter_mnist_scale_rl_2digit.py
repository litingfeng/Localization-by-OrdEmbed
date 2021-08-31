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

    def __init__(self, root, train=True, bg_name=None, transform=None,
                 target_transform=None, sample_size='whole',
                 download=False, digit=4, clutter=0, datapath=None):
        super(MNIST_CoLoc, self).__init__(root, train=train, download=download,
                                          transform=transform, target_transform=target_transform)
        # make sure calling of transform, after which target [x,y,w,h]->[x1,y1,x2,y2]
        assert (self.transform is not None)

        self.digit = digit
        self.datapath = datapath
        self.totensor = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.0334, 0.0301, 0.0279), (0.1653, 0.1532, 0.1490))
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        phase = 'train' if train else 'test'
        datadir = os.path.join(self.datapath, phase)
        if clutter == 1:
            print('loading {} data... '.format(bg_name))
            if 'patch' not in bg_name:
                self.new_digit_data = np.load(os.path.join(datadir, bg_name,
                                                           str(digit) + '_{}_data.npy'.format(bg_name)))  #
                self.new_targets_all = np.load(os.path.join(datadir, bg_name,
                                                        str(digit) + '_{}_label.npy'.format(
                                                            bg_name)))  # '_label.npy'))
            else:
                self.new_digit_data = np.load(os.path.join(datadir, bg_name,
                                                           str(digit) + '_data.npy'))
                self.new_targets_all = np.load(os.path.join(datadir, bg_name,
                                                        str(digit) + '_label.npy'))

        #self.new_targets = torch.from_numpy(self.new_targets)
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

    def __getitem__(self, index):
        img, target, target_noise = self.new_digit_data[index], self.new_targets[index], \
                                    self.new_targets_noise[index]
        target = np.stack((target, target_noise))
        #target = np.stack((target_noise, target))
        img = Image.fromarray(img)#.convert(mode='RGB')

        if DEBUG:
            print('before target ', target)
            f, ax = plt.subplots()
            x1, y1, x2, y2, _ = target[0]
            ax.imshow(np.asarray(img), interpolation='none')
            patch = Rectangle((x1, y1), x2, y2, linewidth=1,
                              edgecolor='g', facecolor='none', fill=False)
            ax.add_patch(patch)
            plt.title('org')
            plt.show()


        if self.transform is not None:
            img, target = self.transform(img, target)

        if DEBUG:
            print('after target ', target)
            f, ax = plt.subplots()
            x1, y1, x2, y2, _ = target[1]
            ax.imshow(np.asarray(img), interpolation='none')
            patch = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,
                              edgecolor='g', facecolor='none', fill=False)
            ax.add_patch(patch)
            plt.show()

        img = self.totensor(img)

        if self.target_transform is not None:
            target_new = self.target_transform(target)
            # print('target_new ', target_new)
            # exit()
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
    from transform import MyBoxScaleTransform
    kwargs = {'num_workers': 0, 'pin_memory': True}

    train_transform = Compose([#RandomHorizontalFlip(),
                               #RandomShear(),
                               #RandomTranslate(),
                               Resize(84)])
    test_transform = Compose([Resize(84)])
    trainset = MNIST_CoLoc(root='.', train=True, digit=3, sample_size='50',
                           bg_name='random_patch_2digit',
                           datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                           clutter=1, transform=train_transform)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=True, **kwargs)

    for i, (data, target) in enumerate(train_loader):
        print(type(target))
        print('data ', data.shape, 'target ', target.shape)
        break




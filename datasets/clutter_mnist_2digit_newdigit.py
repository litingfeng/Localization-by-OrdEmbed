"""
Created on 4/22/2021 10:42 PM

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
    def __init__(self, root, digit, train=True, bg_name=None, transform=None,
                 switch=False, target_transform=None, support_size=5, sample_size='whole',
                 download=False, clutter=0, datapath=None):
        super(MNIST_CoLoc, self).__init__(root, train=train, download=download,
                                          transform=transform,
                                          target_transform=target_transform)

        assert (datapath is not None)
        #assert (digit != 4 and support_size != 0)

        self.switch = switch
        self.digit = digit
        self.datapath = datapath
        self.totensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        phase = 'train' if train else 'test'
        datadir = os.path.join(self.datapath, phase)
        if clutter == 1:
            print('loading {} data... '.format(bg_name))
            if 'patch' not in bg_name:
                self.new_digit_data = np.load(os.path.join(datadir, bg_name,
                                                           str(digit) + '_{}_data.npy'.format(bg_name)))  #
                self.new_targets = np.load(os.path.join(datadir, bg_name,
                                                            str(digit) + '_{}_label.npy'.format(
                                                                bg_name)))  # '_label.npy'))
            else:
                self.new_digit_data = np.load(os.path.join(datadir, bg_name,
                                                           str(digit) + '_data.npy'))
                self.new_targets = np.load(os.path.join(datadir, bg_name,
                                                            str(digit) + '_label.npy'))
        elif clutter == 0:
            print('Not implemented')
            exit()

        # during training, only support_size images in train set has gt box
        if train:
            inds_path = os.path.join(datadir, 'clutter_2digit', str(digit) + '_inds.npy')
            if not os.path.exists(inds_path):
                print(digit, ' inds_path empty, creating...')
                inds = np.random.permutation(len(self.new_digit_data))
                np.save(inds_path, arr=inds)
            inds = np.load(inds_path)
            if sample_size != 'whole':
                sample_size = int(sample_size)
                self.new_digit_data = self.new_digit_data[inds[:sample_size]]
                self.new_targets = self.new_targets[inds[:sample_size]]

            #inds = np.random.permutation(len(self.new_digit_data))
            self.supp_data_array = self.new_digit_data[inds[:support_size]].copy()
            self.supp_targets_all = self.new_targets[inds[:support_size]].copy()

            # convert data to model input
            self.supp_data = []
            for i in range(self.supp_data_array.shape[0]):
                img, target = self.supp_data_array[i], self.supp_targets_all[i]
                img = Image.fromarray(img)
                if self.transform is not None:
                    img, target = self.transform(img, target)
                # transform target to [x1, y1, x2, y2]
                target[:, 2] = target[:, 0] + target[:, 2] - 1
                target[:, 3] = target[:, 1] + target[:, 3] - 1
                img = self.totensor(img)
                self.supp_data.append(img)
            self.supp_data = torch.stack(self.supp_data) # (support_size, 1, 84, 84)
            self.supp_targets_all = torch.from_numpy(self.supp_targets_all) # (support_size, 2, 5)
            if self.switch == 1:
                self.supp_targets = self.supp_targets_all[self.supp_targets_all[:, :, -1] != self.digit][:, :-1]
                self.supp_targets_noise = self.supp_targets_all[self.supp_targets_all[:, :, -1] == self.digit][:, :-1]
            elif self.switch == 0:
                self.supp_targets = self.supp_targets_all[self.supp_targets_all[:, :, -1] == self.digit][:, :-1]
                self.supp_targets_noise = self.supp_targets_all[self.supp_targets_all[:, :, -1] != self.digit][:, :-1]

    def __getitem__(self, index):
        img, target = self.new_digit_data[index], self.new_targets[index]
        img = Image.fromarray(img)
        #print('target ', target)

        if DEBUG:
            print('supp target ', self.supp_targets)
            print('before target ', target)
            f, ax = plt.subplots()
            x1, y1, x2, y2, _ = target[0]
            ax.imshow(np.asarray(img), cmap='gray', interpolation='none')
            patch = Rectangle((x1, y1), x2, y2, linewidth=2,
                              edgecolor='g', facecolor='none', fill=False)
            ax.add_patch(patch)
            plt.title('before')
            plt.show()

        if self.transform is not None:
            img, target = self.transform(img, target)

        if DEBUG:
            print('after target ', target)
            f, ax = plt.subplots()
            x1, y1, x2, y2, _ = target[1]
            ax.imshow(np.asarray(img), cmap='gray', interpolation='none')
            patch = Rectangle((x1, y1), x2, y2, linewidth=2,
                              edgecolor='g', facecolor='none', fill=False)
            ax.add_patch(patch)
            plt.title('after')
            plt.show()

        # transform target to [x1, y1, x2, y2]
        target[:, 2] = target[:, 0] + target[:, 2] - 1
        target[:, 3] = target[:, 1] + target[:, 3] - 1

        img = self.totensor(img)

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

    kwargs = {'num_workers': 0, 'pin_memory': True}

    trainset = MNIST_CoLoc(root='.', train=True, digit=3, switch=0,
                           support_size=5, bg_name='random_patch_2digit_34',
                           datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                           clutter=1)
    print('supp target ', trainset.supp_targets)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=True, **kwargs)

    for i, (data, target) in enumerate(train_loader):
        print(type(target))
        print('data ', data.shape, 'target ', target.shape)
        break


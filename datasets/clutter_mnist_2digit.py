"""
Created on 5/10/2021 7:34 PM

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

class MNIST_CoLoc_2Digit(datasets.MNIST):
    def __init__(self, root, train=True, sample_size='whole',
                 transform=None, target_transform=None,
                 download=False, digit=4, digit2=9, clutter=0, datapath=None):
        super(MNIST_CoLoc_2Digit, self).__init__(root,
                                train=train, download=download,
                                transform=transform,
                                target_transform=target_transform)

        self.digit, self.digit2 = digit, digit2
        self.datapath = datapath
        self.totensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        if self.datapath is None:
            self.digit_data = [idx for idx, digit in enumerate(self.targets)
                               if digit == self.digit]
            print('total number of digit-{} image\t{}'.format(self.digit, len(self.digit_data)))
            self.digit2_data = [idx for idx, digit in enumerate(self.targets)
                               if digit == self.digit2]
            num_digit2 = len(self.digit2_data)
            print('total number of digit-{} image\t{}'.format(self.digit2, num_digit2))
            # generate Translated images, put each 28*28 digit at random location in a 84*84

            # black canvas
            self.new_digit_data = np.zeros((len(self.digit_data), 84, 84), dtype=float)
            # note! target style should be [x,y,w, h]
            self.new_targets = torch.zeros((len(self.digit_data), 2, 5), dtype=float)
            for i, idx in enumerate(self.digit_data):
                # sample a location
                x_d = random.randint(0, 84 - 28)
                y_d = random.randint(0, 84 - 28)
                data = self.data[idx]
                self.new_digit_data[i, y_d:y_d + 28, x_d:x_d + 28] = data
                self.new_targets[i, 0] = torch.tensor([x_d, y_d, 28, 28, self.digit])
                # sample a second random location for the second digit
                data2 = self.data[self.digit2_data[np.random.choice(num_digit2)]]
                done = False
                while not done:
                    x_d2 = random.randint(0, 84 - 28)
                    y_d2 = random.randint(0, 84 - 28)
                    iou = box_iou(torch.Tensor([[x_d, y_d, x_d+28-1, y_d+28-1]]),
                                  torch.Tensor([[x_d2, y_d2, x_d2+28-1, y_d2+28-1]]))
                    if iou == 0: done = True

                self.new_digit_data[i, y_d2:y_d2 + 28, x_d2:x_d2 + 28] = data2
                self.new_targets[i, 1] = torch.tensor([x_d2, y_d2, 28, 28, self.digit2])

                if clutter == 1:
                    for _ in range(32): # 16
                        # crop from noise data
                        noise_data = self.data[random.randint(0,len(self.data)-1)]
                        x0 = random.randint(0, 28 - 6)
                        y0 = random.randint(0, 28 - 6)
                        cropped = noise_data[y0:y0+6, x0:x0+6]
                        # sample a location to put cropped noise data
                        x = random.randint(0, 84 - 6)
                        y = random.randint(0, 84 - 6)
                        while np.sum(self.new_digit_data[i, y:y + 6, x:x + 6]) != 0:
                            x = random.randint(0, 84 - 6)
                            y = random.randint(0, 84 - 6)
                        # Insert digit fragment, but not on top of digits
                        if np.sum(self.new_digit_data[i, y:y + 6, x:x + 6]) == 0:
                            self.new_digit_data[i, y:y + 6, x:x + 6] = cropped

                        # Clip any over-saturated pixels
                        self.new_digit_data[i] = np.clip(self.new_digit_data[i], 0, 255)

        else:
            phase = 'train' if train else 'test'
            datadir = os.path.join(self.datapath, phase)
            if clutter == 1:
                # data with fixed size digit 28*28
                self.new_digit_data = np.load(os.path.join(datadir, 'clutter_2digit',
                                                           str(digit) + '_clutter_2digit_data.npy'))
                self.new_targets = np.load(os.path.join(datadir, 'clutter_2digit',
                                                        str(digit) + '_clutter_2digit_label.npy'))

        if train and sample_size != 'whole':
            # inds = np.random.permutation(len(self.new_digit_data))
            # np.save(os.path.join(datadir, 'moreclutter', str(digit) + '_inds'), arr=inds)
            inds = np.load(os.path.join(datadir, 'moreclutter', str(digit) + '_inds.npy'))
            sample_size = int(sample_size)
            self.new_digit_data = self.new_digit_data[inds[:sample_size]]
            self.new_targets = self.new_targets[inds[:sample_size]]

    def __getitem__(self, index):
        img, target = self.new_digit_data[index], self.new_targets[index]
        img = Image.fromarray(img)

        if DEBUG:
            print('before target 1 ', target[0])
            f, ax = plt.subplots()
            x1, y1, x2, y2, _ = target[0]
            ax.imshow(np.asarray(img), cmap='gray', interpolation='none')
            patch = Rectangle((x1, y1), x2, y2, linewidth=1,
                              edgecolor='g', facecolor='none', fill=False)
            ax.add_patch(patch)
            x1, y1, x2, y2, _ = target[1]
            ax.imshow(np.asarray(img), cmap='gray', interpolation='none')
            patch = Rectangle((x1, y1), x2, y2, linewidth=1,
                              edgecolor='g', facecolor='none', fill=False)
            ax.add_patch(patch)
            plt.title('org')
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

    kwargs = {'num_workers': 0, 'pin_memory': True}
    # generate synthesized image and save
    savepath = '/research/cbim/vast/tl601/Dataset/Synthesis_mnist'
    # for digit in range(0, 10):
    #     done = False
    #     while not done:
    #         digit2 = np.random.choice(10)
    #         if digit != digit2: done = True
    #     for split in [True, False]:
    #         dataset = MNIST_CoLoc_2Digit(root='.', train=split, digit=digit,
    #                                 digit2=digit2,
    #                                 sample_size='whole',
    #                                #datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
    #                                clutter=1)
    #
    #         print(digit, ' ', split, ' digit data ', dataset.new_digit_data.shape,
    #               ' target ', dataset.new_targets.shape)
    #         savedir = 'train' if split else 'test'
    #         savedir = os.path.join(savepath, savedir)
    #         np.save('{}/clutter_2digit/{}_clutter_2digit_data'.format(savedir, digit),
    #                 dataset.new_digit_data)
    #         np.save('{}/clutter_2digit/{}_clutter_2digit_label'.format(savedir, digit), dataset.new_targets)

    dataset = MNIST_CoLoc_2Digit(root='.', train=True, digit=0,
                                 digit2=1,
                                 sample_size='whole',
                                 datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                                 clutter=1)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=12, shuffle=True, **kwargs)
    for i, (data, target) in enumerate(train_loader):
        print('data ', data.shape, ' target ', target.shape)
        exit()
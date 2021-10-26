# -*- coding: utf-8 -*-
# @Time : 9/7/21 3:04 PM
# @Author : Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
'''
Generate corrupted mnist dataset with different background:
clean, random patch, cluttered, gaussian noise, impulse noise
'''

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
        # generate Translated images, put each DIGIT_SIZE*DIGIT_SIZE or more scaled digit
        # at random location in a IMG_SIZE*IMG_SIZE black canvas
        self.new_digit_data = np.zeros((len(self.digit_data), IMG_SIZE, IMG_SIZE), dtype='uint8')
        # note! target style should be [x,y,w, h]
        self.new_targets = torch.zeros((len(self.digit_data), 4), dtype=float)

        for i, idx in enumerate(self.digit_data):
            # sample a location
            x_d = random.randint(0, IMG_SIZE - DIGIT_SIZE)
            y_d = random.randint(0, IMG_SIZE - DIGIT_SIZE)
            data = self.data[idx]

            self.new_digit_data[i, y_d:y_d + DIGIT_SIZE, x_d:x_d + DIGIT_SIZE] = data
            self.new_targets[i] = torch.tensor([x_d, y_d, DIGIT_SIZE, DIGIT_SIZE])

            # add background
            if bg_name != 'clean':
                if bg_name == 'patch':
                    for _ in range(4):
                        # sample a location
                        width = np.random.randint(low=int(DIGIT_SIZE * 0.7), high=int(DIGIT_SIZE * 2.0))
                        x = random.randint(0, IMG_SIZE - width)
                        y = random.randint(0, IMG_SIZE - width)
                        while box_iou(torch.tensor([[x, y, x + width - 1, y + width - 1]]),
                                      torch.tensor([[x_d, y_d, x_d + DIGIT_SIZE - 1, y_d + DIGIT_SIZE - 1]])) > 0.:
                            width = np.random.randint(low=int(DIGIT_SIZE * 0.7), high=int(DIGIT_SIZE * 2.0))
                            x = random.randint(0, IMG_SIZE - width)
                            y = random.randint(0, IMG_SIZE - width)
                        gray = np.ones((width, width), dtype=np.uint8) * \
                               np.random.randint(50, 220)
                        self.new_digit_data[i, y:y + width, x:x + width] = gray
                        # Clip any over-saturated pixels
                        self.new_digit_data[i] = np.clip(self.new_digit_data[i], 0, 255)

                elif bg_name == 'clutter':
                    for _ in range(16):
                        '''generate random noise patches'''
                        # crop from noise data
                        noise_data = self.data[random.randint(0, len(self.data) - 1)]
                        x0 = random.randint(0, DIGIT_SIZE - 6)
                        y0 = random.randint(0, DIGIT_SIZE - 6)
                        cropped = noise_data[y0:y0 + 6, x0:x0 + 6]
                        # sample a location to put cropped noise data
                        x = random.randint(0, IMG_SIZE - 6)
                        y = random.randint(0, IMG_SIZE - 6)
                        while np.sum(self.new_digit_data[i, y:y + 6, x:x + 6]) != 0:
                            x = random.randint(0, IMG_SIZE - 6)
                            y = random.randint(0, IMG_SIZE - 6)
                        # Insert digit fragment, but not on top of digits
                        if np.sum(self.new_digit_data[i, y:y + 6, x:x + 6]) == 0:
                            self.new_digit_data[i, y:y + 6, x:x + 6] = cropped.numpy()
                        # Clip any over-saturated pixels
                        self.new_digit_data[i] = np.clip(self.new_digit_data[i], 0, 255)
                else:
                    if bg_name == 'impulse_noise':
                        self.new_digit_data[i] = torch.from_numpy(round_and_astype(
                            np.array(impulse_noise(self.new_digit_data[i], severity=2))))
                    elif bg_name == 'gaussian_noise':
                        self.new_digit_data[i] = torch.from_numpy(round_and_astype(
                            np.array(gaussian_noise(self.new_digit_data[i], severity=2))))


    def __getitem__(self, index):
        pass

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

    if args.dataset == 'mnist':
        for digit in range(10):
            for split in [True, False]:
                dataset = MNIST_Corrupted(root='/research/cbim/vast/tl601/Dataset',
                                          bg_name=args.bg_name, train=split,
                                          digit=digit, transform=Compose([Resize(IMG_SIZE)]))
                print(digit, ' ', split, ' digit data ', dataset.new_digit_data.shape,
                      ' target ', dataset.new_targets.shape)
                savedir = 'train' if split else 'test'
                savedir = os.path.join(savepath, savedir)
                np.save('{}/{}/{}_data'.format(savedir, args.bg_name, digit), dataset.new_digit_data)
                np.save('{}/{}/{}_label'.format(savedir, args.bg_name, digit), dataset.new_targets)
                #break
            #break

    else:
        print('Not implemented')
        exit()






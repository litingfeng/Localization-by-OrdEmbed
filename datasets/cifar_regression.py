"""
Created on 3/8/2021 12:25 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import copy
import math
import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets

DEBUG = False
class myCIFAR(datasets.CIFAR10):
    def __init__(self, root, datapath=None, train=True, transform=None,
                 target_transform=None, download=True):
        super(myCIFAR, self).__init__(root, train=train, download=download,
                                      transform=transform, target_transform=target_transform)
        phase = 'train' if train else 'test'
        self.angles = transform.angles
        self.cls2idx = {a: i for i,a in enumerate(self.angles)}
        print('cls2idx ', self.cls2idx)

        if not datapath:
            assert (1==0)
            # select 4, 1
            self.newdata, self.newtargets = [], []
            for idx, digit in enumerate(self.targets):
                if digit == 4 or digit == 1:
                    self.newdata.append(idx)
                    target = 0 if digit == 4 else 1
                    self.newtargets.append(target)
            self.newdata, self.newtargets = np.array(self.newdata), \
                                            np.array(self.newtargets)
            pickle.dump((self.newdata, self.newtargets),
                        open('{}_oridinal.pkl'.format(phase), 'wb'))
        else:
            # self.newdata, self.newtargets = \
            #     pickle.load(open('{}_oridinal.pkl'.format(phase), 'rb'))
            self.newdata, self.newtargets, self.rot_inds, self.norot_inds = \
                pickle.load(open('{}_cifar_rot{}_more.pkl'.format(phase,
                                                                  int(0.5 * 100)), 'rb'))

        # use 1-ratio non-rotated samples
        self.data = [self.data[self.newdata[i]] for i in self.norot_inds]
        self.targets = self.newtargets[self.norot_inds]
        self.data = np.stack(self.data)

    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]
        img = Image.fromarray(data)

        if DEBUG:
            f, ax = plt.subplots()
            ax.imshow(np.asarray(img), cmap='gray', interpolation='none')
            plt.title('org')
            plt.show()

        img = self.transform(img)
        a1, a2, img0, img1, img2 = img
        if DEBUG:
            print('a1 ', a1, ' a2 ', a2)
            print('nexp ', img1.size, img2.size)
            f, ax = plt.subplots()
            ax.imshow(np.asarray(img1), cmap='gray', interpolation='none')
            plt.title('img1_nexp')
            plt.show()
            f, ax = plt.subplots()
            ax.imshow(np.asarray(img2), cmap='gray', interpolation='none')
            plt.title('img2_nexp')
            plt.show()
            exit()

        return self.cls2idx[a1], self.cls2idx[a2], \
               img0, img1, img2, target

    def __len__(self):
        return len(self.data)

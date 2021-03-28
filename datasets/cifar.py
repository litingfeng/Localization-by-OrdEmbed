"""
Created on 3/7/2021 3:04 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import copy
import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms.functional as TF

DEBUG = False
class myCIFAR(datasets.CIFAR10):
    def __init__(self, root, subset, ratio=0.3, datapath=None,
                 angles=np.array(range(0, 181, 15)),
                 train=True, transform=None,
                 target_transform=None, download=True):
        super(myCIFAR, self).__init__(root, train=train, download=download,
                                      transform=transform,
                                      target_transform=target_transform)
        self.subset = subset
        self.ratio = ratio
        self.angles = angles
        self.cls2idx = {a: i for i, a in enumerate(self.angles)}
        self.idx2cls = {i: a for i, a in enumerate(self.angles)}
        print('cls2idx ', self.cls2idx)

        phase = 'train' if train else 'test'
        if not datapath:
            # select 4, 1
            self.newdata, self.newtargets = [], []
            for idx, digit in enumerate(self.targets):
                if digit == 4 or digit == 1:
                    self.newdata.append(idx)
                    target = 0 if digit == 4 else 1
                    self.newtargets.append(target)
            self.newdata, self.newtargets = np.array(self.newdata), \
                                        np.array(self.newtargets)
            self.rot_inds = np.random.choice(len(self.newdata),
                                             size=int(self.ratio*len(self.newdata)), replace=False) # select 100 to rotate
            print('rot number of 4 {}\nrot number of 1 {}'.format(
                len(np.where(self.newtargets[self.rot_inds] == 0)[0]),
                len(np.where(self.newtargets[self.rot_inds] == 1)[0])))
            self.norot_inds =np.array(list(set(range(len(self.newdata))) - set(self.rot_inds)))
            print('self.norot_inds ', self.norot_inds.shape)
            pickle.dump((self.newdata, self.newtargets, self.rot_inds, self.norot_inds),
                        open('{}_cifar_rot{}_more.pkl'.format(phase,
                                                              int(self.ratio*100)), 'wb'))
        else:
            self.newdata, self.newtargets, self.rot_inds, self.norot_inds = \
                pickle.load(open('{}_cifar_rot{}_more.pkl'.format(phase,
                                                                  int(self.ratio*100)), 'rb'))
        print('number of 4 {}\nnumber of 1 {}'.format(len(np.where(self.newtargets == 0)[0]),
                                                      len(np.where(self.newtargets == 1)[0])))
        print(' rot: {} norot_inds: {} '.format(self.rot_inds.shape, self.norot_inds.shape))

        # select which subset to train
        if self.subset == 'original_all': # use all original(non-roated) 200 samples
            self.data = [self.data[i] for i in self.newdata]
            self.targets = self.newtargets
            self.data = np.stack(self.data)
            self.targets_ang = np.zeros(len(self.data))
        elif self.subset == 'original_half': # use 1-ratio non-rotated samples
            print('self.norot_inds ', self.norot_inds.shape)
            self.data = [self.data[self.newdata[i]] for i in self.norot_inds]
            self.targets = self.newtargets[self.norot_inds]
            self.data = np.stack(self.data)
        elif self.subset == 'half_half':
            self.orgdata = [self.data[i] for i in self.newdata] # HWC
            self.data = copy.deepcopy(self.orgdata)
            self.targets_ang = np.zeros(len(self.data))
            for i, inds in enumerate(self.rot_inds):
                img = Image.fromarray(self.data[inds])
                angle = np.random.choice(self.angles)
                img = TF.rotate(img, float(angle))
                self.data[inds] = np.array(img)
                self.targets_ang[inds] = angle
            self.data = np.stack(self.data) # (10000, 32, 32, 3)
            self.targets = self.newtargets
        else:
            print('Not implementd')
            exit()

        print('subset [{}] data: {}'.format(self.subset, self.data.shape[0]))

        if self.subset == 'half_half':
            self.four_rot, self.one_rot = [], []
            for i in self.rot_inds:
                self.four_rot.append(i) if self.targets[i] == 0 \
                    else self.one_rot.append(i)
            self.four_norot, self.one_norot = [], []
            for i in self.norot_inds:
                self.four_norot.append(i) if self.targets[i] == 0 \
                    else self.one_norot.append(i)
            print('rot 4: {} rot 1: {}'.format(
                len(self.four_rot), len(self.one_rot)))
            print('nonrot 4: {} nonrot 1: {}'.format(
                len(self.four_norot), len(self.one_norot)))

    def __getitem__(self, index):
        data, target, target_a = self.data[index], self.targets[index], \
                                 self.targets_ang[index]
        img = Image.fromarray(data)

        if DEBUG:
            for i in (10, 1200, 2200):
                inds = self.rot_inds[i]
                data, target, target_a = self.data[inds], self.targets[inds], \
                                         self.targets_ang[inds]
                img = Image.fromarray(data)
                f, ax = plt.subplots()
                ax.imshow(np.asarray(
                    Image.fromarray(self.orgdata[inds])),
                    cmap='gray', interpolation='none')
                plt.title('org')
                print('target ', target)
                plt.show()

                f, ax = plt.subplots()
                ax.imshow(np.asarray(img), cmap='gray', interpolation='none')
                plt.title('trans angle {:.2f}'.format(target_a))
                plt.show()
            exit()
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.cls2idx[target_a], target_a, index

    def __len__(self):
        return len(self.data)
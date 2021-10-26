# -*- coding: utf-8 -*-
# @Time : 9/7/21 3:00 PM
# @Author : Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
import os
import torch
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from util.utils import box_iou

#TODO 2 digit
DEBUG = False

class MNIST_Corrupted(datasets.MNIST):
    def __init__(self, root, train, bg_name, digit, datapath, support_size=5, anchors=None,
                 sample_size='whole', transform=None, target_transform=None, download=False):
        super(MNIST_Corrupted, self).__init__(root, train=train, download=download,
                                              transform=transform, target_transform=target_transform)

        ''' make sure calling of transform, after which target [x,y,w,h]->[x1,y1,x2,y2] '''
        assert (self.transform is not None)

        self.digit    = digit
        self.totensor = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))])
        phase         = 'train' if train else 'test'
        data_dir      = os.path.join(datapath, phase)

        ''' load data '''
        self.new_digit_data = np.load(os.path.join(data_dir, bg_name,
                                                   str(digit) + '_data.npy'))
        self.new_targets    = np.load(os.path.join(data_dir, bg_name,
                                                   str(digit) + '_label.npy'))

        self.new_targets = torch.from_numpy(self.new_targets)

        if train and sample_size != 'whole':
            inds_file = os.path.join(data_dir, 'sample_inds', str(digit) + '_inds.npy')
            if not os.path.isfile(inds_file):
                inds = np.random.permutation(len(self.new_digit_data))
                np.save(os.path.join(data_dir, 'sample_inds', str(digit) + '_inds'), arr=inds)
            else:
                print('loading sample inds file...')
                inds = np.load(inds_file)
            sample_size = int(sample_size)
            self.new_digit_data = self.new_digit_data[inds[:sample_size]]
            self.new_targets = self.new_targets[inds[:sample_size]]

        ''' compute ious between anchors and gt '''
        self.anchors = None
        if anchors is not None: # pretrain
            self.anchors = anchors
            # transform target format
            new_targets = self.new_targets.clone()
            new_targets[:,2] = new_targets[:, 2] + new_targets[:, 0] - 1
            new_targets[:, 3] = new_targets[:, 3] + new_targets[:, 1] - 1
            # compute ious & iou_map. iou_map is generated using group random sampling
            self.ious = box_iou(new_targets, anchors)
            self.iou_map = [{i: None for i in range(10)} for _ in range(new_targets.shape[0])]
            for i in range(new_targets.shape[0]):
                for j in range(10):
                    inds = torch.where((self.ious[i]>=j*0.1) & (self.ious[i] < (j+1)*0.1 ))[0]
                    if len(inds) != 0 :
                        self.iou_map[i][j] = inds
                    else:
                        del self.iou_map[i][j]

            # if DEBUG:
            #     print('num anchors ', anchors.shape)
            #     print('targets ', new_targets[0], self.new_targets[0])
            #     print('self.iou_map[0][3] ', self.iou_map[0][3])
            #     print(self.ious[0, self.iou_map[0][3]])
            #     for j in range(10):
            #         if j in self.iou_map[0].keys():
            #             print(len(self.iou_map[0][j]))
            #         else:
            #             print(j)
            #     exit()
        elif train: # adapt
            inds = np.random.permutation(len(self.new_digit_data))
            if sample_size != 'whole':
                sample_size = int(sample_size)
                self.new_digit_data = self.new_digit_data[inds[:sample_size]]
                self.new_targets = self.new_targets[inds[:sample_size]]
            self.supp_data_array = self.new_digit_data[inds[:support_size]].copy()
            self.supp_targets = self.new_targets[inds[:support_size]].clone()

            # convert data to model input
            self.supp_data = []
            for i in range(self.supp_data_array.shape[0]):
                img, target = self.supp_data_array[i], self.supp_targets[i].reshape(1, 4)
                img = Image.fromarray(img)
                if self.transform is not None:
                    img, target = self.transform(img, target)
                img = self.totensor(img)
                self.supp_data.append(img)
            self.supp_data = torch.stack(self.supp_data)  # (support_size, 1, 84, 84)

    def __getitem__(self, index):
        img, target = self.new_digit_data[index], self.new_targets[index].numpy().reshape(1, 4)
        if self.anchors is not None: # pretrain
            iou_map, ious = self.iou_map[index], self.ious[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img, target = self.transform(img, target)

        if DEBUG:
            f, ax = plt.subplots()
            x1, y1, x2, y2 = target[0]
            print('new target ', target)
            ax.imshow(np.asarray(img), cmap='gray', interpolation='none')
            patch = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,
                              edgecolor='g', facecolor='none', fill=False)
            ax.add_patch(patch)

        img = self.totensor(img)

        if self.anchors is not None: # pretrain
            '''random sample two ious and two box with these ious'''
            keys = list(iou_map.keys())
            smp_iou = np.random.choice(keys, size=2)
            ind1, ind2 = np.random.choice(iou_map[smp_iou[0]]), \
                         np.random.choice(iou_map[smp_iou[1]])
            ti, tj, ioui, iouj = self.anchors[ind1], self.anchors[ind2], \
                                 ious[ind1], ious[ind2]

            if DEBUG:
                print('ti ', ti, ' tj ', tj)
                print('ious ', ioui, ' ', iouj)
                patch = Rectangle((ti[0], ti[1]), ti[2]-ti[0], ti[3]-ti[1], linewidth=1,
                                  edgecolor='r', facecolor='none', fill=False)
                ax.add_patch(patch)
                patch = Rectangle((tj[0], tj[1]), tj[2] - tj[0], tj[3] - tj[1], linewidth=1,
                                  edgecolor='y', facecolor='none', fill=False)
                ax.add_patch(patch)
                plt.show()
                exit()
            #print('target new ', target)
            return img, target.flatten(), ti, tj, ioui, iouj
        else:
            return img, target.flatten()

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
    from util.utils import generate_boxes

    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_transform = Compose([Resize(84),
                               #RandomHorizontalFlip(),
                               ])
    test_transform = Compose([Resize(84)])
    anchors = generate_boxes(base_size=4,
                             feat_height=21, feat_width=21, img_size=84,
                             feat_stride=4,
                             ratios=[1.0],
                             min_box_side=28,
                             scales=np.array(range(7, 20)))

    print('number of anchors for 84*84 image ', anchors.shape[0])
    print('anchors ', anchors, torch.min(anchors[:,2:]))
    trainset = MNIST_Corrupted(root='.', train=True, digit=3,
                               bg_name='clutter',
                               datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist_github',
                               transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=512, shuffle=True, **kwargs)

    #for i, (data, target, ti, tj, ioui, iouj ) in enumerate(train_loader):
    for i, (data, target) in enumerate(train_loader):
        print('target ', target.shape)
        break
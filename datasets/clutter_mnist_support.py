"""
Created on 4/25/2021 4:38 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import torch
import numpy as np
from util.utils import box_iou
from torchvision import datasets, transforms

DEBUG = False
class MNIST_Supp():
    def __init__(self, supp_data, supp_targets, anchors=None):
        super(MNIST_Supp, self).__init__()

        self.anchors = anchors
        self.supp_data = supp_data.clone()
        self.supp_targets = supp_targets.clone()
        if anchors is not None:
            # for each support image, compute iou(gt, all anchors)
            supp_targets_copy = supp_targets.clone()
            self.ious = box_iou(supp_targets_copy, anchors)
            # for each image, group according to iou, group then save in dict
            self.iou_map = [{i: None for i in range(10)}
                            for _ in range(supp_targets_copy.shape[0])]
            for i in range(supp_targets_copy.shape[0]):
                for j in range(10):
                    inds = torch.where((self.ious[i] >= j * 0.1) &
                                       (self.ious[i] < (j + 1) * 0.1))[0]
                    if len(inds) != 0:
                        self.iou_map[i][j] = inds
                    else:
                        del self.iou_map[i][j]

            if DEBUG:
                print('num anchors ', anchors.shape)
                print('targets ', supp_targets_copy[0], supp_targets[0])
                print('self.iou_map[0][3] ', self.iou_map[0][3])
                print(self.ious[0, self.iou_map[0][3]])
                for j in range(10):
                    if j in self.iou_map[0].keys():
                        print(len(self.iou_map[0][j]))
                    else:
                        print(j)
                #exit()

    def __getitem__(self, index):
        img, target = self.supp_data[index], self.supp_targets[index]
        #iou_map, ious = self.iou_map[index], self.ious[index]

        # # random sample two ious and two box with these ious
        # keys = list(iou_map.keys())
        # smp_iou = np.random.choice(keys, size=2)
        # ind1, ind2 = np.random.choice(iou_map[smp_iou[0]]), \
        #              np.random.choice(iou_map[smp_iou[1]])
        # ti, tj, iou1, iou2 = self.anchors[ind1], self.anchors[ind2], \
        #                      ious[ind1], ious[ind2]

        return img, target#, ti, tj, iou1, iou2

    def __len__(self):
        return self.supp_data.shape[0]

if __name__ == '__main__':
    import torch
    from util.data_aug import *
    from util.augmentations import Compose
    from util.utils import generate_boxes
    from datasets.clutter_mnist_newdigit import MNIST_CoLoc

    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_transform = Compose([Resize(84)])
    test_transform = Compose([Resize(84)])
    anchors = generate_boxes(base_size=4,
                             feat_height=21, feat_width=21, img_size=84,
                             feat_stride=4,
                             ratios=[1.0],
                             min_box_side=28,
                             scales=np.array(range(7, 20)))
    print('number of anchors for 84*84 image ', anchors.shape[0])

    trainset = MNIST_CoLoc(root='.', train=True, digit=5,
                           support_size=5, sample_size='whole',
                           datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                           clutter=1, transform=train_transform)
    suppset = MNIST_Supp(supp_data=trainset.supp_data,
                         supp_targets=trainset.supp_targets,
                         anchors=anchors)
    supp_loader = torch.utils.data.DataLoader(
        trainset, batch_size=512, shuffle=True, **kwargs)
    print('support image ', len(suppset.supp_data))

    for i, (data, target, ti, tj, iou1, iou2) in enumerate(supp_loader):
        print('data ', data.shape)
        print('ti ', ti, '\ntj ', tj, '\niou1 ', iou1, '\niou2 ', iou2)
        break
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
from scipy.ndimage.interpolation import zoom
from torchvision import datasets, transforms
from util.utils import box_iou

DEBUG = False

class MNIST_CoLoc(datasets.MNIST):

    def __init__(self, root, train=True, anchors=None,
                 transform=None, target_transform=None,
                 download=False, digit=4, clutter=0, datapath=None):
        super(MNIST_CoLoc, self).__init__(root, train=train, download=download,
                                          transform=transform, target_transform=target_transform)
        # make sure calling of transform, after which target [x,y,w,h]->[x1,y1,x2,y2]
        assert (self.transform is not None)

        self.digit = digit
        self.datapath = datapath
        self.totensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        if self.datapath is None:
            self.digit_data = [idx for idx, digit in enumerate(self.targets)
                               if digit == self.digit]
            print('total number of digit-{} image\t{}'.format(self.digit, len(self.digit_data)))

            self.new_digit_data = np.zeros((len(self.digit_data), 84, 84), dtype=float)
            # note! target style should be [x,y,w, h]
            self.new_targets = torch.zeros((len(self.digit_data), 4), dtype=float)
            for i, idx in enumerate(self.digit_data):
                data = self.data[idx]
                # Randomly Scale image
                h = np.random.randint(low=28, high=int(28 * 2.5))
                w = np.random.randint(low=28, high=int(28 * 2.5))
                data = zoom(data, (h / 28., w / 28.))

                # sample a location
                x_d = random.randint(0, 84 - w)
                y_d = random.randint(0, 84 - h)

                self.new_digit_data[i, y_d:y_d + h, x_d:x_d + w] = data

                # Assemble bounding box
                gt_bbox = self.create_gt_bbox(self.new_digit_data[i], 20)

                self.new_targets[i] = torch.tensor(gt_bbox)
                # add clutter if possible
                if clutter == 1:
                    for _ in range(32):  # 16
                        # crop from noise data
                        noise_data = self.data[random.randint(0, len(self.data) - 1)]
                        x0 = random.randint(0, 28 - 6)
                        y0 = random.randint(0, 28 - 6)
                        cropped = noise_data[y0:y0 + 6, x0:x0 + 6]
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
                # different scale and aspect ratio
                self.new_digit_data = np.load(os.path.join(datadir, 'clutterScaleARLarger',
                                                           str(digit) + '_clutterScaleARLarger_data.npy'))
                self.new_targets = np.load(os.path.join(datadir, 'clutterScaleARLarger',
                                                        str(digit) + '_clutterScaleARLarger_label.npy'))
            elif clutter == 0:
                print('Not implemented')
                exit()

        self.new_targets = torch.from_numpy(self.new_targets)
        self.anchors = anchors
        # for each image, compute iou(gt, all anchors)
        new_targets = self.new_targets.clone()
        new_targets[:,2] = new_targets[:, 2] + new_targets[:, 0] - 1
        new_targets[:, 3] = new_targets[:, 3] + new_targets[:, 1] - 1
        self.ious = box_iou(new_targets, anchors)

        self.iou_map = [{i: None for i in range(10)} for _ in range(new_targets.shape[0])]
        for i in range(new_targets.shape[0]):
            for j in range(10):
                inds = torch.where((self.ious[i]>=j*0.1) & (self.ious[i] < (j+1)*0.1 ))[0]
                if len(inds) != 0 :
                    self.iou_map[i][j] = inds
                else:
                    del self.iou_map[i][j]
        if DEBUG:
            print('num anchors ', anchors.shape)
            print('targets ', new_targets[0], self.new_targets[0])
            print('self.iou_map[0][3] ', self.iou_map[0][3])
            print(self.ious[0, self.iou_map[0][3]])
            for j in range(10):
                if j in self.iou_map[0].keys():
                    print(len(self.iou_map[0][j]))
                else:
                    print(j)
            exit()

    def __getitem__(self, index):
        img, target = self.new_digit_data[index], self.new_targets[index]
        iou_map, ious = self.iou_map[index], self.ious[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img, target = self.transform(img, target)

        if DEBUG:
            f, ax = plt.subplots()
            x1, y1, x2, y2 = target
            ax.imshow(np.asarray(img), cmap='gray', interpolation='none')
            patch = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,
                              edgecolor='g', facecolor='none', fill=False)
            ax.add_patch(patch)

        img = self.totensor(img)

        # random sample two ious and two box with these ious
        keys = list(iou_map.keys())
        smp_iou = np.random.choice(keys, size=2)
        ind1, ind2 = np.random.choice(iou_map[smp_iou[0]]), \
                     np.random.choice(iou_map[smp_iou[1]])
        ti, tj, iou1, iou2 = self.anchors[ind1], self.anchors[ind2], \
                             ious[ind1], ious[ind2]

        if DEBUG:
            print('ti ', ti, ' tj ', tj)
            print('ious ', iou1, ' ', iou2)
            patch = Rectangle((ti[0], ti[1]), ti[2]-ti[0], ti[3]-ti[1], linewidth=1,
                              edgecolor='r', facecolor='none', fill=False)
            ax.add_patch(patch)
            patch = Rectangle((tj[0], tj[1]), tj[2] - tj[0], tj[3] - tj[1], linewidth=1,
                              edgecolor='y', facecolor='none', fill=False)
            ax.add_patch(patch)
            plt.show()
            exit()
        #print('target new ', target)
        return img, target, ti, tj, iou1, iou2

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

    anchors = generate_boxes(base_size=4,
                             feat_height=21, feat_width=21, img_size=84,
                             feat_stride=4,
                             ratios=np.linspace(0.5, 3.0, num=10),
                             min_box_side=28,
                             scales=np.array(range(7, 20)))
    print('number of anchors for 84*84 image ', anchors.shape[0])

    # generate synthesized image and save
    savepath = '/research/cbim/vast/tl601/Dataset/Synthesis_mnist'
    for digit in range(0, 10):
        for split in [True, False]:
            dataset = MNIST_CoLoc(root='.', clutter=True, train=split, digit=digit,
                                  anchors=anchors,
                                  transform=Compose([Resize(84)]))
            print(digit, ' ', split, ' digit data ', dataset.new_digit_data.shape,
                  ' target ', dataset.new_targets.shape)
            savedir = 'train' if split else 'test'
            savedir = os.path.join(savepath, savedir)
            np.save('{}/clutterScaleARLarger/{}_clutterScaleARLarger_data'.format(savedir, digit),
                    dataset.new_digit_data)
            np.save('{}/clutterScaleARLarger/{}_clutterScaleARLarger_label'.format(savedir, digit), dataset.new_targets)



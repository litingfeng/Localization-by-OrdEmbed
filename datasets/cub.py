"""
min_box_side: 25,
aspect ratio: 0.15~4
Created on 3/24/2021 1:54 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
"""
Created on 10/7/2020 1:14 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import os
import numpy as np
import torchvision
import pandas as pd
from torchvision import transforms
from PIL import Image, ImageDraw
from util.utils import convert_image_np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
#matplotlib.use('agg')
from util.augmentations import Compose, resize
from util.utils import box_iou
import torch

DEBUG = False

class CUB_OneClass():
    def __init__(self, root, category, anchors=None, transform=None):
        self.transform = transform
        self.totensor = transforms.ToTensor()
        if type(category) not in (tuple, list):
            category = (category,)

        boxes = np.loadtxt(os.path.join(root, 'CUB_200_2011', 'bounding_boxes.txt'))
        images = pd.read_csv(os.path.join(root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        images.set_index("img_id", inplace=True)

        # get image ids of this category
        image_class_labels = pd.read_csv(os.path.join(root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        thiscls = image_class_labels.loc[image_class_labels['target'].isin(category)]
        this_id = thiscls['img_id'].to_numpy()

        this_image = images.loc[this_id]
        self.target = boxes[this_id-1, 1:]
        #convert target[x, y, w, h]->[x1, y1, x2, y2]
        self.target[:,2] = self.target[:,0] + self.target[:,2] - 1
        self.target[:, 3] = self.target[:, 1] + self.target[:, 3] - 1
        # print('self.target ', self.target.shape, np.min(self.target[:, 2:]))
        # exit()

        self.data = []
        for i, row in this_image.iterrows():
            path = os.path.join(root, 'CUB_200_2011/images', row['filepath'])
            img = Image.open(path)
            if img.getbands()[0] == 'L':
                img = img.convert('RGB')
            self.data.append(img.copy())
            img.close()

        #print('category ', category)
        print('number of images {}'.format(len(self.data)))

        self.anchors = anchors
        ''' for each image, compute iou(gt, all anchors)'''
        target = torch.from_numpy(self.target.copy())
        # convert target to 224 images,
        for i in range(target.shape[0]):
            img = self.data[i]
            width, height = img.size
            target[i, 0] /= width
            target[i, 2] /= width
            target[i, 1] /= height
            target[i, 3] /= height
        target = target* 224
        self.ious = box_iou(target, anchors)

        self.iou_map = [{i: None for i in range(10)} for _ in range(target.shape[0])]
        for i in range(target.shape[0]):
            for j in range(10):
                inds = torch.where((self.ious[i] >= j * 0.1) & (self.ious[i] < (j + 1) * 0.1))[0]
                if len(inds) != 0:
                    self.iou_map[i][j] = inds
                else:
                    del self.iou_map[i][j]

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        iou_map, ious = self.iou_map[index], self.ious[index]
        if self.transform is not None:
                img, target, _ = self.transform(img, np.expand_dims(target, axis=0))
        if DEBUG:
            print('after img ', img.shape)
            print(img.min(), img.max())
            img_ = convert_image_np(torchvision.utils.make_grid(img.unsqueeze(0)),
                                    norm=True)
            #draw = ImageDraw.Draw(img)
            box = target * 224
            print('box new ', box)
            # draw.rectangle(((box[0], box[1]),
            #                 (box[2], box[3])), outline="red")
            # # self.data[idx].show()
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(img_)
            x1, y1, x2, y2 = box[0]
            patch = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,
                              edgecolor='r', facecolor='none', fill=False)
            ax.add_patch(patch)
            #plt.savefig('img_trns.png')
            plt.show()

        # random sample two ious and two box with these ious
        keys = list(iou_map.keys())
        smp_iou = np.random.choice(keys, size=2)
        # print('smp iou ', smp_iou)
        ind1, ind2 = np.random.choice(iou_map[smp_iou[0]]), \
                     np.random.choice(iou_map[smp_iou[1]])
        # print('inds12 ', ind1, ind2)
        ti, tj, iou1, iou2 = self.anchors[ind1], self.anchors[ind2], \
                             ious[ind1], ious[ind2]

        return img, np.squeeze(target)* 224, ti, tj, iou1, iou2

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    import torch
    from util.data_aug import *
    from util.bbox_util import *
    from util import joint_transforms as t
    from util.utils import generate_boxes
    from constants import VOC_ENCODING, VOC_DECODING, \
        IMAGENET_MEAN, IMAGENET_STD
    import torch.nn.functional as F

    anchors = generate_boxes(base_size=16, feat_height=14, feat_width=14,
                             min_box_side=25, img_size=224-1,
                             feat_stride=16, ratios=[0.5, 1.0, 2.0, 3.0, 4.0],
                             scales=np.array([1.5, 3.0, 6.0, 9.0, 12.0]))

    print('number of anchors for 224*224 image ', anchors.shape[0])
    print('anchors ', anchors)

    cubdataset = CUB_OneClass('/research/cbim/vast/tl601/Dataset',
                              category=list(range(1,2)),
                              anchors=anchors,
                              transform=t.Compose([
                t.ConvertFromPIL(),
                t.ToPercentCoords(),
                t.Resize(224),
                t.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                t.ToTensor() # no change to (0, 1)
            ]))
    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(cubdataset, batch_size=1, shuffle=True, **kwargs)

    data, target = next(iter(train_loader))
    print('data ', data.shape, '\ttarget ', target.shape)

    ################# plot histogram of statistics
    # min_w , min_h = 224, 224
    # targets = []
    # new_regions = []
    # box_side, aspect_ratio = [], []
    # for i, (data, target) in enumerate(train_loader):
    #     target = target.int()
    #     wh = target[:, 2:] - target[:, :2] # (bs, 2)
    #     ar = wh[:,0] / wh[:, 1] # (bs, )
    #     wh = wh.view(-1) # (2bs,)
    #     box_side += [wh.numpy()]
    #     aspect_ratio += [ar.numpy()]
    #
    # box_side = np.concatenate(box_side)
    # aspect_ratio = np.concatenate(aspect_ratio)
    # print(box_side.shape, aspect_ratio.shape)
    #
    #
    # fig, [ax1, ax2] = plt.subplots(1, 2)
    # n, bins, patches = ax1.hist(box_side, 50, density=True, facecolor='g')
    # ax1.set_title('Histogram of box side')
    #
    # n, bins, patches = ax2.hist(aspect_ratio, 50, density=True, facecolor='b')
    # ax2.set_title('Histogram of aspect ratio')
    # fig.suptitle('CUB Dataset Histograms')
    # #plt.grid(True)
    # #plt.show()
    # plt.savefig('cub_statistics.png')




"""
Created on 5/11/2021 9:59 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import os
import numpy as np
from torchvision import transforms
import json
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import transforms
from PIL import Image, ImageDraw
from util.utils import convert_image_np
from matplotlib.patches import Rectangle
from util.augmentations import Compose, resize
from util.utils import box_iou
import torch
import xml.etree.ElementTree as ET
from datasets.batch_sampler import PrototypicalBatchSampler

DEBUG = False

class CUB_fewshot():
    def __init__(self, root, img_size, bg_name=None, mode='base', support_size=5, transform=None):
        self.transform = transform
        self.img_size = img_size

        boxes = np.loadtxt(os.path.join(root, 'bounding_boxes.txt'))
        images = pd.read_csv(os.path.join(root, 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])

        images.set_index("filepath", inplace=True)
        if mode == 'base':
            data_file = os.path.join(root, 'fewshot', 'warbler_train15.json')
        else:
            if bg_name == 'warbler':
                datalist = 'warbler_val5.json'
            elif bg_name == 'gull':
                datalist = 'gull_59_64.json'
            elif bg_name == 'wren':
                datalist = 'wren_193_197.json'
            elif bg_name == 'sparrow':
                datalist = 'sparrow_114_119.json'
            elif bg_name == 'oriole':
                datalist  = 'oriole_95_98.json'
            elif bg_name == 'kingfisher':
                datalist = 'kingfisher_79_83.json'
            elif bg_name == 'vireo':
                datalist = 'vireo_151_157.json'
            else:
                print('background {} not implemented'.format(bg_name))
                exit()
            print('loading {}...'.format(datalist))
            data_file = os.path.join(root, 'fewshot', datalist)
            #data_file = os.path.join(root, 'fewshot', 'warbler_val5.json'.format(mode))
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        print("{} images {} labels {}".format(mode, len(self.meta['image_names']),
              len(set(self.meta['image_labels']))))
        self.data, self.target = [], []
        self.this_cate = [i for i, idx in enumerate(self.meta['image_labels']) if idx == 0]
        # print('self ', len(self.this_cate))
        # print('labels ', set(self.meta['image_labels']))
        # exit()
        for i in range(len(self.meta['image_names'])):
            image_path = os.path.join(self.meta['image_names'][i])
            img = Image.open(image_path)
            if img.getbands()[0] == 'L':
                img = img.convert('RGB')
            self.data.append(img.copy())
            img.close()

            this_id = images.loc[image_path.split('images/')[1]]
            self.target.append(boxes[this_id-1, 1:].flatten())

        self.target = np.vstack(self.target)
        # convert target[x, y, w, h]->[x1, y1, x2, y2]
        self.target[:, 2] = self.target[:, 0] + self.target[:, 2] - 1
        self.target[:, 3] = self.target[:, 1] + self.target[:, 3] - 1

        # sample support data
        inds = np.random.permutation(len(self.data))[:support_size]
        self.supp_data_array = [self.data[i].copy() for i in inds]
        self.supp_targets_ = self.target[inds[:support_size]].copy()

        # convert data to model input
        self.supp_data, self.supp_targets = [], np.zeros((support_size, 4))
        for i in range(len(self.supp_data_array)):
            img, target = self.supp_data_array[i], self.supp_targets_[i]
            if self.transform is not None:
                img, target, _ = self.transform(img, np.expand_dims(target, axis=0))
            target = (np.squeeze(target) ) * self.img_size
            self.supp_data.append(img)
            self.supp_targets[i] = target
        self.supp_data = torch.stack(self.supp_data)  # (support_size, 1, 84, 84)
        self.supp_targets = torch.from_numpy(self.supp_targets)

    def read_content(self, xml_file: str):

        tree = ET.parse(xml_file)
        root = tree.getroot()

        list_with_all_boxes = []

        for boxes in root.iter('object'):
            filename = root.find('filename').text

            ymin, xmin, ymax, xmax = None, None, None, None

            ymin = int(boxes.find("bndbox/ymin").text)
            xmin = int(boxes.find("bndbox/xmin").text)
            ymax = int(boxes.find("bndbox/ymax").text)
            xmax = int(boxes.find("bndbox/xmax").text)

            list_with_single_boxes = [xmin, ymin, xmax, ymax]
            list_with_all_boxes.append(list_with_single_boxes)

        return filename, list_with_all_boxes

    def __getitem__(self, index):
        #index = self.this_cate[index]
        img, target, label = self.data[index], self.target[index].copy(), \
                             self.meta['image_labels'][index]

        if self.transform is not None:
            img, target, _ = self.transform(img, np.expand_dims(target, axis=0))

        if DEBUG:
            print('index ', index, ' target ', target, '\n',
                  np.hstack((np.squeeze(target) * self.img_size, np.array([label])))
                  )
            print(self.meta['image_names'][index])
            print('after img ', img.shape)
            print(img.min(), img.max())
            img_ = convert_image_np(torchvision.utils.make_grid(img.unsqueeze(0)),
                                    norm=True)
            # draw = ImageDraw.Draw(img)
            box = target * self.img_size
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
            plt.show()
            # plt.savefig('img_trns.png')

        return img, np.hstack((np.squeeze(target) * self.img_size))

    def __len__(self):
        #return len(self.this_cate)
        return len(self.data)

def init_sampler(classes_per_set, samples_per_class, labels, iterations, mode):
    if 'train' in mode:
        classes_per_it = classes_per_set
        num_samples = samples_per_class
    else:
        classes_per_it = classes_per_set
        num_samples = samples_per_class

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=iterations)

if __name__ =='__main__':
    from util.utils import generate_boxes
    from util.data_aug import *
    from util.bbox_util import *
    from util import joint_transforms as t
    from constants import IMAGENET_MEAN, IMAGENET_STD

    dataset = CUB_fewshot('/research/cbim/vast/tl601/Dataset/CUB_200_2011',
                          img_size=224, mode='base', support_size=5,
                          transform=t.Compose([
                              t.ConvertFromPIL(),
                              t.ToPercentCoords(),
                              t.Resize(224),
                              t.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                              t.ToTensor()  # no change to (0, 1)
                          ])
                          )
    train_batchsampler = init_sampler(5, 5, dataset.meta['image_labels'], 200, 'train')
    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=10, shuffle=True, **kwargs)
        #batch_sampler=train_batchsampler)
    print('supp_target ', dataset.supp_targets)


    ars = []
    ws, hs = [], []
    min_w , min_h = 224, 224
    targets = []
    new_regions = []
    box_side, aspect_ratio = [], []
    for i, (data, target, _) in enumerate(train_loader):
        print(target)
        exit()
        target = target.int()
        wh = target[:, 2:4] - target[:, :2] # (bs, 2)
        ar = wh[:,0] / wh[:, 1] # (bs, )
        wh = wh.view(-1) # (2bs,)
        box_side += [wh.numpy()]
        aspect_ratio += [ar.numpy()]

    box_side = np.concatenate(box_side)
    aspect_ratio = np.concatenate(aspect_ratio)
    print(box_side.shape, aspect_ratio.shape)


    fig, [ax1, ax2] = plt.subplots(1, 2)
    n, bins, patches = ax1.hist(box_side, 50, density=True, facecolor='g')
    ax1.set_title('Histogram of box side')

    n, bins, patches = ax2.hist(aspect_ratio, 50, density=True, facecolor='b')
    ax2.set_title('Histogram of aspect ratio')
    fig.suptitle('CUB Dataset Histograms')
    #plt.grid(True)
    plt.show()
    # plt.savefig('cub_statistics.png')


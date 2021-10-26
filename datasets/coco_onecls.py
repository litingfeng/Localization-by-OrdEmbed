"""
Created on 9/25/2021 5:54 PM
@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import torchvision
from util.utils import convert_image_np
from PIL import Image
from shutil import copyfile
import matplotlib.pyplot as plt
from collections import Counter
from pycocotools.coco import COCO
from matplotlib.patches import Rectangle
import numpy as np
from util.utils import box_iou
import os
import os.path
import torch
from torchvision.datasets.vision import VisionDataset

DEBUG = False

class CocoDataset(VisionDataset):
    def __init__(self, root, annFile, anchors=None, img_size=224,
                 selected_cls=['cat'], support_size=5, transform = None,
                 target_transform = None, transforms = None):
        super().__init__(root, transforms, transform, target_transform)
        assert (len(selected_cls) == 1)

        self.selected_cls = selected_cls
        self.coco = COCO(annFile)
        self.img_size = img_size
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.phase = 'train' if 'train' in annFile else 'val'

        filelist = '/research/cbim/vast/tl601/Dataset/coco/subsets/{}_{}.txt'.format(self.phase, selected_cls[0])
        self.catIds = self.coco.getCatIds(catNms=selected_cls)
        if not os.path.isfile(filelist):
            assert (anchors is None)  # confirm in pretrain stage
            self._select_categories()
            with open(filelist, 'w') as f:
                f.write('\n'.join([str(id) for id in self.imgIds]))
        else:
            with open(filelist, 'r') as f:
                self.imgIds = f.read().splitlines()
            self.imgIds = [int(id) for id in self.imgIds]

        print("Number of images containing {}: {}".format(selected_cls[0], len(self.imgIds)))

        '''load targets'''
        self.targets = []
        for id in self.imgIds:
            target = [t['bbox']+[t['category_id']] for t in self._load_target(id) if t['category_id'] in self.catIds]
            assert (len(target) == 1)
            self.targets.append(np.array(target))
        self.targets = np.vstack(self.targets)[:, :4]
        # convert target[x, y, w, h]->[x1, y1, x2, y2]
        self.targets[:, 2] = self.targets[:, 0] + self.targets[:, 2] - 1
        self.targets[:, 3] = self.targets[:, 1] + self.targets[:, 3] - 1
        ''''''

        self.anchors = None
        if anchors is not None: # pretrain
            self.anchors = anchors
            self.ious, self.iou_maps = [], []
            '''convert targets to specified size'''
            target_224 = self.targets.copy()
            for i in range(target_224.shape[0]):
                img = self._load_image(self.imgIds[i])
                width, height = img.size
                target_224[i, 0] /= width
                target_224[i, 2] /= width
                target_224[i, 1] /= height
                target_224[i, 3] /= height
            target_224 = target_224 * self.img_size

            '''generate pairs for objects'''
            new_targets = torch.from_numpy(target_224.copy().reshape(-1, 4))
            self.ious = box_iou(new_targets, anchors)
            # for each image, sort according to iou, group then save in dict
            self.iou_map = [{i: None for i in range(10)} for _ in range(new_targets.shape[0])]
            for i in range(new_targets.shape[0]):
                for j in range(10):
                    inds = torch.where((self.ious[i] >= j * 0.1) & (self.ious[i] < (j + 1) * 0.1))[0]
                    if len(inds) != 0:
                        self.iou_map[i][j] = inds
                    else:
                        del self.iou_map[i][j]

            if DEBUG:
                print('num anchors ', anchors.shape)
                #print('targets ', target_224[0], self.target[0])
                print('self.iou_map[0][3] ', self.iou_map[0][3])
                print(self.ious[0, self.iou_map[0][3]])
                for j in range(10):
                    if j in self.iou_map[0].keys():
                        print(len(self.iou_map[0][j]))
                    else:
                        print(j)
                #exit()
        else: # adaptation
            '''load support data and transform to model input format'''
            if self.phase == 'train':
                self.supp_data, self.supp_target = [], []
                inds = np.random.permutation(len(self.imgIds))
                self.supp_ids = np.array(self.imgIds)[inds[:support_size]]
                shuf_targets = self.targets[inds[:support_size]]
                for i, id in enumerate(self.supp_ids):
                    image, target = self._load_image(int(id)), shuf_targets[i].reshape(1, 4)
                    if self.transforms is not None:
                        image, target, _ = self.transforms(image, target)
                    target = target * self.img_size  # make sure calling ToPercentCoords() in transform
                    self.supp_data.append(image)
                    self.supp_target.append(target)

                self.supp_data = torch.stack(self.supp_data)  # (support_size, 3, 224, 224)
                self.supp_targets = torch.from_numpy(np.stack(self.supp_target)).squeeze()  # (support_size, 2, 5)


    def _select_categories(self):
        '''select images that have all selected_cls, and only one instance for each class in an image'''
        self.catIds, all_ids = [], []
        for sel_cls in self.selected_cls:
            this_catIds = self.coco.getCatIds(catNms=sel_cls)
            this_imgIds = self.coco.catToImgs[this_catIds[0]]
            this_imgIds = [id for id, num in Counter(this_imgIds).items() if num == 1]
            all_ids.append(set(this_imgIds))
            self.catIds += this_catIds

        # Get all images containing the above Category IDs
        self.imgIds = list(set.intersection(*all_ids))


    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def _sample_pair(self, iou_map, ious):
        # random sample two ious and two box with these ious
        keys = list(iou_map.keys())
        smp_iou = np.random.choice(keys, size=2)
        ind1, ind2 = np.random.choice(iou_map[smp_iou[0]]), \
                     np.random.choice(iou_map[smp_iou[1]])
        boxi, boxj, ioui, iouj = self.anchors[ind1], self.anchors[ind2], \
                             ious[ind1], ious[ind2]
        return boxi, boxj, ioui, iouj

    def __getitem__(self, index: int):
        id = self.imgIds[index]
        image, target = self._load_image(id), self.targets[index].reshape(1,4)
        if self.anchors is not None:  # pretrain
            iou_map, iou = self.iou_map[index], self.ious[index]

        if self.transforms is not None:
            image, target, _ = self.transforms(image, target)
            target = target * self.img_size  # make sure calling ToPercentCoords() in transform

        if self.anchors is not None:  # pretrain
            boxi, boxj, ioui, iouj = self._sample_pair(iou_map, iou)
            return image, target.flatten(), boxi, boxj, ioui, iouj
        else:
            return image, target.flatten()

    def __len__(self) -> int:
        #return len(self.ids)
        return len(self.imgIds)


if __name__ == '__main__':
    from util.utils import generate_boxes
    from util import joint_transforms as t

    transform = t.Compose([
        t.ConvertFromPIL(),
        t.ToPercentCoords(),
        t.Resize(224),
        t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        t.ToTensor()  # no change to (0, 1)
    ])
    coco_train = CocoDataset(root='/research/cbim/vast/tl601/Dataset/coco/train2017',
                            annFile='/research/cbim/vast/tl601/Dataset/coco/annotations/instances_train2017.json',
                             selected_cls=['cat'],
                             anchors=generate_boxes(base_size=16, feat_height=14, feat_width=14,
                                                    min_box_side=25, img_size=224,
                                                    feat_stride=16, ratios=np.linspace(0.3, 3.5, num=15),
                                                    scales=np.array(range(2, 13))),
                             transforms=transform
                             )

    print('Number of samples: ', len(coco_train))
    train_loader = torch.utils.data.DataLoader(
                    coco_train, batch_size=2, shuffle=True, num_workers=0,
                    pin_memory=True)
    data, target, boxi, boxj, ioui, iouj = next(iter(train_loader))
    print('data ', data.shape, '\ttarget ', target.shape)
    print('boxi ', boxi, '\nboxj ', boxj)
    print('ioui ', ioui, '\niouj ', iouj)




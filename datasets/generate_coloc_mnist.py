"""
dataloader for MNIST co-localization experiment
Supplementary
Created on 10/29/2020 8:28 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import os
import torch
from webcolors import name_to_rgb
from PIL import Image
from util.utils import box_iou
import random
import numpy as np
from util.corruptions import impulse_noise, fog, gaussian_noise
from scipy.ndimage.interpolation import zoom
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

DEBUG = False
# 16 colors
colors = ['aqua', 'blue', 'fuchsia', 'gray', 'green',
          'lime', 'maroon', 'navy', 'olive', 'orange', 'purple',
          'red', 'silver', 'teal', 'white', 'yellow']
def round_and_astype(x):
    return np.round(x).astype(np.uint8)
# def color_grayscale_arr(arr, red=True):
#   """Converts grayscale image to either red or green"""
#   assert arr.ndim == 2
#   dtype = arr.dtype
#   h, w = arr.shape
#   arr = np.reshape(arr, [h, w, 1])
#   if red:
#     arr = np.concatenate([arr,
#                           np.zeros((h, w, 2), dtype=dtype)], axis=2)
#   else:
#     arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
#                           arr,
#                           np.zeros((h, w, 1), dtype=dtype)], axis=2)
#   return arr

def color_grayscale_arr(arr, col2rgb=None):
  """Converts grayscale image to random color"""
  assert arr.ndim == 2
  dtype = arr.dtype
  h, w = arr.shape
  inds = (arr > 0)
  new_arr = np.zeros((h,w,3), dtype=dtype)
  if col2rgb is None:
      # random generate color
      col_name = np.random.randint(0, 16)
      col2rgb = name_to_rgb(colors[col_name])
      #print(colors[col_name])
  new_arr[inds, 0] = col2rgb.red
  new_arr[inds, 1] = col2rgb.green
  new_arr[inds, 2] = col2rgb.blue

  return new_arr, col2rgb

class MNIST_CoLoc(datasets.MNIST):

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, digit=4, clutter=False, datapath=None):
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
        phase = 'train' if train else 'test'
        datadir = os.path.join('/research/cbim/vast/tl601/Dataset/Synthesis_mnist', phase)
        self.clutter_data = np.load(os.path.join(datadir, 'moreclutter',
                                                 str(digit) + '_moreclutter_data.npy'))
        self.clutter_targets = np.load(os.path.join(datadir, 'moreclutter',
                                                    str(digit) + '_moreclutter_label.npy'))
        if self.datapath is None:
            self.digit_data = [idx for idx, digit in enumerate(self.targets)
                               if digit == self.digit]
            print('total number of digit-{} image\t{}'.format(self.digit, len(self.digit_data)))
            # generate Translated images, put each 28*28 or more scaled digit at random location in a 84*84
            # black canvas
            self.new_digit_data = np.zeros((len(self.digit_data), 84, 84), dtype='uint8')
            # note! target style should be [x,y,w, h]
            self.new_targets = torch.zeros((len(self.digit_data), 4), dtype=float)
            for i, idx in enumerate(self.digit_data):
                # location same as cluttered image
                # x_d = int(self.clutter_targets[i, 0])
                # y_d = int(self.clutter_targets[i, 1])
                # sample a location
                # x_d = random.randint(0, 84 - 28)
                # y_d = random.randint(0, 84 - 28)
                # 5 pixels margin for enlarge
                x_d = random.randint(5, 84 - 38)
                y_d = random.randint(5, 84 - 38)
                # x_d = random.randint(14, 42) # at least 1/4 digit in in the center box
                # y_d = random.randint(14, 42)
                data = self.data[idx]
                # new_data, rgb = color_grayscale_arr(data.numpy())
                # f, ax = plt.subplots()
                # ax.imshow(data.numpy(), cmap='gray', interpolation='none')
                # plt.title('org')
                # plt.show()
                # f, ax = plt.subplots()
                # ax.imshow(new_data, interpolation='none')
                # plt.title('color')
                # plt.show()
                # exit()

                self.new_digit_data[i, y_d:y_d + 28, x_d:x_d + 28] = data

                # 5 pixels margin for enlarge, true target (x_d,y_d, 28,28)
                self.new_targets[i] = torch.tensor([x_d-5, y_d-5, 38, 38])
                #self.new_targets[i] = torch.tensor([x_d, y_d, 28, 28])

                # add clutter if possible
                if clutter == 1:
                    for _ in range(4):  # 16
                        '''generate random pixel patches'''
                        # sample a location
                        width = np.random.randint(low=int(28 * 0.7), high=int(28 * 2.0))
                        x = random.randint(0, 84 - width)
                        y = random.randint(0, 84 - width)
                        while box_iou(torch.tensor([[x, y, x + width - 1, y + width - 1]]),
                                      torch.tensor([[x_d, y_d, x_d + 28 - 1, y_d + 28 - 1]])) > 0.:
                            width = np.random.randint(low=int(28 * 0.7), high=int(28 * 2.0))
                            x = random.randint(0, 84 - width)
                            y = random.randint(0, 84 - width)
                        gray = np.ones((width, width), dtype=np.uint8) * \
                               np.random.randint(50, 220)
                        self.new_digit_data[i, y:y + width, x:x + width] = gray
                        '''generate random noise patches'''
                        # # crop from noise data
                        # noise_data = self.data[random.randint(0, len(self.data) - 1)]
                        # x0 = random.randint(0, 28 - 6)
                        # y0 = random.randint(0, 28 - 6)
                        # cropped = noise_data[y0:y0 + 6, x0:x0 + 6]
                        # # sample a location to put cropped noise data
                        # x = random.randint(0, 84 - 6)
                        # y = random.randint(0, 84 - 6)
                        # while np.sum(self.new_digit_data[i, y:y + 6, x:x + 6, 0]) != 0:
                        #     x = random.randint(0, 84 - 6)
                        #     y = random.randint(0, 84 - 6)
                        # # Insert digit fragment, but not on top of digits
                        # if np.sum(self.new_digit_data[i, y:y + 6, x:x + 6, 0]) == 0:
                        #     self.new_digit_data[i, y:y + 6, x:x + 6, :] = color_grayscale_arr(cropped.numpy(), rgb)[0]

                        # Clip any over-saturated pixels
                        self.new_digit_data[i] = np.clip(self.new_digit_data[i], 0, 255)

                    # impulse noise
                    # self.new_digit_data[i] = torch.from_numpy(round_and_astype(
                    #     np.array(impulse_noise(self.new_digit_data[i], severity=2))))

        else:
            phase = 'train' if train else 'test'
            datadir = os.path.join(self.datapath, phase)
            if clutter == 1:
                self.new_digit_data = np.load(os.path.join(datadir, name, str(digit) + '_data.npy'))
                self.new_targets = np.load(os.path.join(datadir, name, str(digit) + '_label.npy'))
            elif clutter == 0:
                self.new_digit_data = np.load(os.path.join(datadir, str(digit)+'_data.npy'))
                self.new_targets = np.load(os.path.join(datadir, str(digit) + '_label.npy'))

    def create_gt_bbox(self, image, minimum_dim):
        # Tighten box
        rows = np.sum(image, axis=0).round(1)
        cols = np.sum(image, axis=1).round(1)

        left = np.nonzero(rows)[0][0]
        right = np.nonzero(rows)[0][-1]
        upper = np.nonzero(cols)[0][0]
        lower = np.nonzero(cols)[0][-1]

        # If box is too narrow or too short, pad it out to >12
        width = right - left
        if width < minimum_dim:
            pad = np.ceil((minimum_dim - width) / 2)
            left = int(left - pad)
            right = int(right + pad)

        height = lower - upper
        if height < minimum_dim:
            pad = np.ceil((minimum_dim - height) / 2)
            upper = int(upper - pad)
            lower = int(lower + pad)

        gt_bbox = [int(left), int(upper), int(right)-int(left)+1, int(lower)-int(upper)+1]

        return gt_bbox

    def __getitem__(self, index):
        img, img_clutter, target, target_clutter = self.new_digit_data[index], self.clutter_data[index],\
                                   self.new_targets[index], self.clutter_targets[index]
        img, img_clutter = Image.fromarray(img), Image.fromarray(img_clutter)

        if self.transform is not None:
            img, target = self.transform(img, target)
            img_clutter, target_clutter = self.transform(img_clutter, target_clutter)

        if DEBUG:

            x1, y1, x2, y2 = target
            print('box ', x1, y1, x2, y2)
            print('target_clutter ', target_clutter)
            f, ax = plt.subplots()
            ax.imshow(np.asarray(img), cmap='gray', interpolation='none')
            patch = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,
                              edgecolor='r', facecolor='none', fill=False)
            ax.add_patch(patch)
            patch = Rectangle((x1+5, y1+5), x2 - x1-10, y2 - y1-10, linewidth=2,
                              edgecolor='y', facecolor='none', fill=False)
            ax.add_patch(patch)
            plt.title('red')
            plt.show()
            f, ax = plt.subplots()
            ax.imshow(np.asarray(img_clutter), cmap='gray',interpolation='none')
            patch = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,
                              edgecolor='r', facecolor='none', fill=False)
            ax.add_patch(patch)
            plt.show()

        img = self.totensor(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
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
    from util.data_aug import *
    from util.augmentations import Compose

    # generate synthesized image and save
    savepath = '/research/cbim/vast/tl601/Dataset/Synthesis_mnist'
    name = 'random_patch_loose'
    os.makedirs(os.path.join(savepath, 'train', name), exist_ok=True)
    os.makedirs(os.path.join(savepath, 'test', name), exist_ok=True)
    for digit in range(0,10):
        for split in [True, False]:
            dataset = MNIST_CoLoc(root='.', clutter=1, train=split, digit=digit,
                                  transform = Compose([Resize(84)]))
            print(digit, ' ', split, ' digit data ', dataset.new_digit_data.shape,
                  ' target ', dataset.new_targets.shape)
            savedir = 'train' if split else 'test'
            savedir = os.path.join(savepath, savedir)
            np.save('{}/{}/{}_data'.format(savedir, name, digit), dataset.new_digit_data)
            np.save('{}/{}/{}_label'.format(savedir, name, digit), dataset.new_targets)
        #     break
        # break

    # min_ar, max_ar = 100, 0
    # for digit in range(10):
    #     dataset = MNIST_CoLoc(root='.', clutter=True, train=True, digit=digit,
    #                           transform=Compose([Resize(84)]),
    #                           datapath=savepath)
    #     data_loader = torch.utils.data.DataLoader(dataset,
    #                   batch_size=1, shuffle=True, num_workers=0)
    #
    #     for i, (data, target ) in enumerate(data_loader):
    #         #print('data ', data.shape, ' target ', target)
    #         w, h = target[:,2] - target[:,0], target[:, -1] - target[:,1]
    #         ar = h / w
    #         min_ar = min(min_ar, torch.min(ar).item())
    #         max_ar = max(max_ar, torch.max(ar).item())
    #         break
    #     break
    #
    # print('min ', min_ar, ' max ', max_ar)

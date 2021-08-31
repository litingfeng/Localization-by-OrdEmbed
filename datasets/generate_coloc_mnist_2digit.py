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

DEBUG = True
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
                 download=False, digit=4, digit2=9, clutter=0, datapath=None):
        super(MNIST_CoLoc, self).__init__(root, train=train, download=download,
                                          transform=transform, target_transform=target_transform)
        # make sure calling of transform, after which target [x,y,w,h]->[x1,y1,x2,y2]
        assert (self.transform is not None)


        self.digit2, self.digit = digit2, digit
        self.datapath = datapath
        self.totensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        phase = 'train' if train else 'test'
        datadir = os.path.join('/research/cbim/vast/tl601/Dataset/Synthesis_mnist', phase)
        if self.datapath is None:
            self.digit_data = [idx for idx, digit in enumerate(self.targets)
                               if digit == self.digit]
            print('total number of digit-{} image\t{}'.format(self.digit, len(self.digit_data)))
            self.digit2_data = [idx for idx, digit in enumerate(self.targets)
                                if digit == self.digit2]
            num_digit2 = len(self.digit2_data)
            print('total number of digit-{} image\t{}'.format(self.digit2, num_digit2))
            # generate Translated images, put each 28*28 or more scaled digit at random location in a 84*84
            # black canvas
            self.new_digit_data = np.zeros((len(self.digit_data), 84, 84), dtype='uint8')
            # note! target style should be [x,y,w, h]
            self.new_targets = torch.zeros((len(self.digit_data),2, 5), dtype=float)
            for i, idx in enumerate(self.digit_data):
                # sample a location
                x_d = random.randint(0, 84 - 28)
                y_d = random.randint(0, 84 - 28)
                data = self.data[idx]
                self.new_digit_data[i, y_d:y_d + 28, x_d:x_d + 28] = data
                self.new_targets[i, 0] = torch.tensor([x_d, y_d, 28, 28, self.digit])

                # sample a second random location for the second digit
                data2 = self.data[self.digit2_data[np.random.choice(num_digit2)]]
                done = False
                while not done:
                    x_d2 = random.randint(0, 84 - 28)
                    y_d2 = random.randint(0, 84 - 28)
                    iou = box_iou(torch.Tensor([[x_d, y_d, x_d + 28 - 1, y_d + 28 - 1]]),
                                  torch.Tensor([[x_d2, y_d2, x_d2 + 28 - 1, y_d2 + 28 - 1]]))
                    if iou == 0: done = True

                self.new_digit_data[i, y_d2:y_d2 + 28, x_d2:x_d2 + 28] = data2
                self.new_targets[i, 1] = torch.tensor([x_d2, y_d2, 28, 28, self.digit2])

                # add clutter if possible
                if clutter == 1:
                    for _ in range(3):  # 16
                        '''generate random pixel patches'''
                        # sample a location
                        width = np.random.randint(low=int(28 * 0.7), high=int(28 * 1.3))
                        x = random.randint(0, 84 - width)
                        y = random.randint(0, 84 - width)
                        while box_iou(torch.tensor([[x, y, x + width - 1, y + width - 1]]),
                                      torch.tensor([[x_d, y_d, x_d + 28 - 1, y_d + 28 - 1]])) > 0. or \
                            box_iou(torch.tensor([[x, y, x + width - 1, y + width - 1]]),
                                    torch.tensor([[x_d2, y_d2, x_d2 + 28 - 1, y_d2 + 28 - 1]])) > 0.:
                            width = np.random.randint(low=int(28 * 0.7), high=int(28 * 1.3))
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

    def __getitem__(self, index):
        img, target = self.new_digit_data[index], self.new_targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img, target = self.transform(img, target)
        if DEBUG:

            x1, y1, x2, y2, _ = target[0]
            print('target ', target)
            f, ax = plt.subplots()
            ax.imshow(np.asarray(img), cmap='gray', interpolation='none')
            patch = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,
                              edgecolor='r', facecolor='none', fill=False)
            ax.add_patch(patch)
            x1, y1, x2, y2,_ = target[1]
            patch = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              edgecolor='y', facecolor='none', fill=False)
            ax.add_patch(patch)
            plt.title('red')
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
    name = 'random_patch_2digit_34'
    os.makedirs(os.path.join(savepath, 'train', name), exist_ok=True)
    os.makedirs(os.path.join(savepath, 'test', name), exist_ok=True)
    for digit in range(3,10):
        done = False
        while not done:
            digit2 = np.random.choice(10)
            if digit != digit2: done = True
        for split in [True, False]:
            dataset = MNIST_CoLoc(root='.', clutter=1, train=split, digit=digit,
                                  digit2=4,
                                  transform = Compose([Resize(84)]))
            print(digit, ' ', split, ' digit data ', dataset.new_digit_data.shape,
                  ' target ', dataset.new_targets.shape)
            savedir = 'train' if split else 'test'
            savedir = os.path.join(savepath, savedir)
            np.save('{}/{}/{}_data'.format(savedir, name, digit), dataset.new_digit_data)
            np.save('{}/{}/{}_label'.format(savedir, name, digit), dataset.new_targets)
            #break
        break

    min_ar, max_ar = 100, 0
    for digit in range(3, 10):
        dataset = MNIST_CoLoc(root='.', clutter=True, train=True, digit=digit,
                              transform=Compose([Resize(84)]),
                              datapath=savepath)
        data_loader = torch.utils.data.DataLoader(dataset,
                      batch_size=1, shuffle=True, num_workers=0)

        for i, (data, target ) in enumerate(data_loader):
            #print('data ', data.shape, ' target ', target)
            w, h = target[:,2] - target[:,0], target[:, -1] - target[:,1]
            ar = h / w
            min_ar = min(min_ar, torch.min(ar).item())
            max_ar = max(max_ar, torch.max(ar).item())
            break
        break

    print('min ', min_ar, ' max ', max_ar)

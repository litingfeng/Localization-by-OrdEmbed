"""
Created on 2/19/2021 12:59 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import numpy as np
import torch
from transform import UnNormalize
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

ROTATE = 0
STAY = 1

#TFs = {0: -15, 1: 15, 2: 0}
TFs = {0: 0, 1: -15}
#TFs = {0: 0, 1: -90}

DEBUG = False
unorm = UnNormalize(mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010))

class Actor:
    def __init__(self):
        pass
    def takeaction(self, data, actions):
        # data (bs, 3, 32, 32), actions (bs,)
        actions = actions.squeeze()
        imgs = []
        for img, action in zip(data, actions):
            if DEBUG:
                img_o = unorm(img.clone())
                img_o = transforms.ToPILImage()(img_o)

            img = TF.rotate(img, TFs[action.item()])
            if DEBUG:
                img_t = unorm(img)
                print('action ', TFs[action.item()])
                img_t = transforms.ToPILImage()(img_t)
                f, ax = plt.subplots()
                ax.imshow(np.asarray(img_o), interpolation='none')
                plt.title('img_o')
                plt.show()
                f, ax = plt.subplots()
                ax.imshow(np.asarray(img_t),  interpolation='none')
                plt.title('img_t')
                plt.show()
                exit()

            imgs.append(img)
        imgs = torch.stack(imgs)
        return imgs

if __name__ == "__main__":
    data = torch.tensor([[[2, 0],
                         [1, 0]],
                        [[2, 5],
                         [1, 5]],
                        [[1, 2],
                         [3, 4]]])
    actions = torch.tensor([[0], [1], [0]])
    print('data ', data, '\nactions ', actions)
    actor = Actor()
    print('trans\n', actor.takeaction(data, actions))
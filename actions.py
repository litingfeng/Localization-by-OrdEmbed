"""
Created on 2/19/2021 12:59 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import numpy as np
import torch
ROTATE = 0
STAY = 1

class Actor:
    def __init__(self):
        pass
    def takeaction(self, data, actions):
        # data (bs, 1, 28, 28), actions (bs,)
        actions = actions.squeeze()
        rot_inds = (actions == ROTATE).nonzero().flatten()
        if len(rot_inds) != 0:
            data[rot_inds] = torch.rot90(data[rot_inds], dims=(-1,-2)) # clockwise
        else:
            print('No data needed to be rotated back')
        return data

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
"""
Created on 3/11/2021 12:40 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import sys
sys.path.append('/research/cbim/vast/tl601/projects/faster-rcnn.pytorch/lib')
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.utils import sample
from model.roi_layers import ROIPool, ROIAlign

DEBUG = False

class EmbeddingNet(nn.Module):
    def __init__(self, norm=1):
        super(EmbeddingNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.norm = norm

    def forward(self, x):
        bs = x.shape[0]
        x0 = F.relu(F.max_pool2d(self.conv1(x), 2)) #(bs, 10, 12, 12)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x0)), 2))
        #x = F.relu(F.max_pool2d(self.conv2(x0), 2))
        x1 = x.view(-1, 320) #(bs, 320)
        x = F.relu(self.fc1(x1))
        x2 = F.dropout(x, training=self.training)
        if self.norm == 1:
            x0 = F.normalize(x0.view(bs, -1), p=2, dim=1)
            x1 = F.normalize(x1, p=2, dim=1)
            x2 = F.normalize(x2, p = 2, dim = 1)
            x = torch.cat((x0, x1, x2), 1) # (bs, 1810)
        else:
            x = torch.cat((x0.view(bs, -1), x1, x2), 1) # (bs, 1810)

        return x

class Net(nn.Module):
    def __init__(self, embedding_net=None, pooling_size=None,
                 pooling_mode=None):
        super(Net, self).__init__()

        self.embedding_net = embedding_net
        self.pooling_mode = pooling_mode

        self.roi_pool = ROIPool((pooling_size, pooling_size), 1.0)
        self.roi_align = ROIAlign((pooling_size, pooling_size),
                                  1.0, -1)  # TODO

    def forward(self, x, rois):
        if self.pooling_mode == 'align':
            embed = self.roi_align(x, rois.view(-1, 5))
        elif self.pooling_mode == 'pool':
            embed = self.roi_pool(x, rois.view(-1, 5))

        embed = self.embedding_net(embed)  # (bs, 1, 28, 28) --> (bs, 1810)

        return embed

class Agent(nn.Module):
    def __init__(self, dim=1810, num_class=2, hidden_size=None):
        super(Agent, self).__init__()
        dim_fc = dim if not hidden_size else hidden_size
        self.fc = nn.Sequential(
            nn.Linear(dim_fc, num_class)
        )
        self.value_head = nn.Linear(dim_fc, 1)

        if hidden_size:
            self.i2h = nn.Linear(dim, hidden_size)
            self.h2h = nn.Linear(hidden_size, hidden_size)

    def forward(self, embed, h_t_prev=None):
        # run through rnn
        if h_t_prev is not None:
            h1 = self.i2h(embed)  # (bs, hidden_size)
            h2 = self.h2h(h_t_prev)  # (bs, hidden_size)
            h_t = F.relu(h1 + h2)  # produce h_t # (bs, hidden_size)

            logits = self.fc(h_t)
            draw = sample(F.softmax(logits, dim=1))
            state_values = self.value_head(h_t.detach())
            return h_t, logits, draw, state_values

        else:
            logits = self.fc(embed)
            draw = sample(F.softmax(logits, dim=1))
            state_values = self.value_head(embed.detach())
            return logits, draw, state_values

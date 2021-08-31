"""
Created on 3/16/2021 11:32 AM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import sys
sys.path.append('/research/cbim/vast/tl601/projects/faster-rcnn.pytorch/lib')
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.roi_layers import ROIPool, ROIAlign
from util.utils import sample

DEBUG = False

class Net(nn.Module):
    def __init__(self, pooling_size=None,
                 pooling_mode=None, init_weights: bool = True):
        super(Net, self).__init__()

        self.pooling_size = pooling_size
        self.RCNN_base = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, stride=2),
            nn.Conv2d(10, 32, kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            #nn.Conv2d(128, 128, kernel_size=3, stride=1),
        )
        if pooling_mode == 'pool':
            self.RCNN_pooler = ROIPool((pooling_size, pooling_size), 1.0 / 5.6)
        elif pooling_mode == 'align':
            self.RCNN_pooler = ROIAlign((pooling_size, pooling_size),
                                       1.0 / 5.6, 2)  # TODO

        self.RCNN_top = nn.Sequential(
            nn.Linear(128 *pooling_size * pooling_size, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 512),
        )
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _head_to_tail(self, pooled):
        pooled_flat = pooled.view(pooled.size(0), -1)
        out = self.RCNN_top(pooled_flat)
        return out

    def forward(self, x, rois):
        self.base_feat = self.RCNN_base(x) # (bs, 64, 15, 15)
        pooled_feat = self.RCNN_pooler(self.base_feat, rois.view(-1,5)) # (bs, 64, 7, 7)
        pooled_feat = self._head_to_tail(pooled_feat) # (bs, 1024)
        return pooled_feat

    def get_embedding(self, x, rois):
        self.base_feat = self.RCNN_base(x)  # (bs, 64, 15, 15)
        pooled_feat = self.RCNN_pooler(self.base_feat, rois.view(-1, 5))  # (bs, 64, 7, 7)
        pooled_feat = self._head_to_tail(pooled_feat)  # (bs, 1024)
        return pooled_feat

class Agent(nn.Module):
    def __init__(self, rnn, dim=1810, num_class=2, hidden_size=None,
                 hist_len=None,
                 init_weights: bool = True):
        super(Agent, self).__init__()
        dim_fc = dim if not hidden_size else hidden_size
        if rnn == 1: #TODO
            self.fc = nn.Sequential(
                nn.Linear(dim_fc, num_class)
            )
            self.i2h = nn.Linear(dim, hidden_size)
            self.h2h = nn.Linear(hidden_size, hidden_size)
        else:
            if hist_len is not None:
                fc_dim = dim + hist_len * num_class
            else:
                fc_dim = dim
            self.fc = nn.Sequential(
                nn.Linear(fc_dim, hidden_size),
                nn.ReLU(True),
                nn.Dropout(p=0.75),
                nn.Linear(48, hidden_size),
                nn.ReLU(True),
                nn.Dropout(p=0.5),
                nn.Linear(hidden_size, num_class)
            )
        # if init_weights:
        #     self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, embed, h_t_prev=None):
        # run through rnn
        if h_t_prev is not None:
            h1 = self.i2h(embed)  # (bs, hidden_size)
            h2 = self.h2h(h_t_prev)  # (bs, hidden_size)
            h_t = F.relu(h1 + h2)  # produce h_t # (bs, hidden_size)

            logits = self.fc(h_t)
            return h_t, logits

        else:
            logits = self.fc(embed)
            return logits
# class Agent(nn.Module):
#     def __init__(self, dim=128, hidden=48, history=0,
#                  hist_len=None, num_class=2):
#         super(Agent, self).__init__()
#         self.dim = dim
#         self.hist = history
#         self.conv = nn.Sequential(
#             nn.Conv2d(128, 32, kernel_size=1, stride=1),
#             nn.ReLU()
#         )
#         if self.hist == 1:
#             fc_dim = 32 * dim * dim + hist_len*num_class
#         else:
#             fc_dim = 32 * dim * dim
#         self.fc = nn.Sequential(
#             nn.Linear(fc_dim, hidden),
#             nn.ReLU(True),
#             nn.Dropout(p=0.75),
#             nn.Linear(hidden, num_class),
#         )
#
#         #init params
#         #init_params(self.fc2)
#
#     def forward(self, embed):
#         #embed = embed.view(embed.shape[0], -1)
#         bs = embed.shape[0]
#         if self.hist == 1:
#             embed, action_history = embed[:, :(self.dim*self.dim*64)].view(-1, 64, self.dim,
#                                                                            self.dim), \
#                                     embed[:, (self.dim*self.dim*64):]
#
#         embed = self.conv(embed).view(embed.shape[0], -1)
#         if self.hist == 1:
#             embed = torch.cat((embed, action_history), 1)
#         logits = self.fc(embed)
#         return logits

if __name__ == '__main__':
    import torch

    net = Net(pooling_mode='align', pooling_size=7)
    model = Agent(dim=1024, hidden_size=24)
    input = torch.rand(3,1,84,84)
    rois = torch.tensor([[0, 42, 42, 83, 83],
                         [1, 28, 28, 56, 56],
                         [2, 14, 14, 83, 83]], dtype=torch.float32) # 41, 29, 70
    output = net(input, rois)

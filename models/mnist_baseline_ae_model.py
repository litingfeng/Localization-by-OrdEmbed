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

class EmbeddingNet(nn.Module):
    def __init__(self, pooling_size=None,
                 pooling_mode=None):
        super(EmbeddingNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        )

        if pooling_mode == 'pool':
            self.roi_pooler = ROIPool((pooling_size, pooling_size), 1.0 / 2.0)
        elif pooling_mode == 'align':
            self.roi_pooler = ROIAlign((pooling_size, pooling_size),
                                        1.0/ 2., 2)

    def get_embedding(self, x, rois):
        base_feat = self.encoder(x)  # (bs, 64, 42, 42)
        embed = self.roi_pooler(base_feat, rois.view(-1, 5))  # (bs, 64, pooling_size, pooling_size)
        #embed = embed.view(embed.size(0), -1)

        return embed

class Agent(nn.Module):
    def __init__(self, dim=128, hidden=48, history=0,
                 hist_len=None, num_class=2):
        super(Agent, self).__init__()
        self.dim = dim
        self.hist = history
        self.conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, stride=1),
            nn.ReLU()
        )
        if self.hist == 1:
            fc_dim = 32 * dim * dim + hist_len*num_class
        else:
            fc_dim = 32 * dim * dim
        self.fc = nn.Sequential(
            nn.Linear(fc_dim, hidden),
            nn.ReLU(True),
            nn.Dropout(p=0.75),
            nn.Linear(hidden, num_class),
        )

        #init params
        #init_params(self.fc2)

    def forward(self, embed):
        #embed = embed.view(embed.shape[0], -1)
        bs = embed.shape[0]
        if self.hist == 1:
            embed, action_history = embed[:, :(self.dim*self.dim*64)].view(-1, 64, self.dim,
                                                                           self.dim), \
                                    embed[:, (self.dim*self.dim*64):]

        embed = self.conv(embed).view(embed.shape[0], -1)
        if self.hist == 1:
            embed = torch.cat((embed, action_history), 1)
        logits = self.fc(embed)
        return logits

if __name__ == '__main__':
    import torch

    embed_net = EmbeddingNet(pooling_size=7, pooling_mode='align')
    model = Agent(dim=7, hidden=24, num_class=10)
    input = torch.rand(3,1,84,84)
    rois = torch.tensor([[0, 42, 42, 83, 83],
                         [1, 28, 28, 56, 56],
                         [2, 14, 14, 83, 83]], dtype=torch.float32) # 41, 29, 70
    output = embed_net.get_embedding(input, rois)
    logits = model(output)
    print(output.shape)
    print(logits.shape, logits.max(1)[1])

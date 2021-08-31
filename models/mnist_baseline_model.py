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
    def __init__(self, norm=True, pooling_size=None,
                 pooling_mode=None):
        super(EmbeddingNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)

        self.norm = norm
        if pooling_mode == 'pool':
            self.roi_pooler = ROIPool((pooling_size, pooling_size), 1.0)
        elif pooling_mode == 'align':
            self.roi_pooler = ROIAlign((pooling_size, pooling_size),
                                        1.0, 2)

    def forward(self, x):
        bs = x.shape[0]
        x0 = F.relu(F.max_pool2d(self.conv1(x), 2)) #(bs, 10, 12, 12)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x0)), 2))
        x1 = x.view(-1, 320) #(bs, 320)
        x = F.relu(self.fc1(x1))
        x2 = F.dropout(x, training=self.training)
        if self.norm:
            x0 = F.normalize(x0.view(bs, -1), p=2, dim=1)
            x1 = F.normalize(x1, p=2, dim=1)
            x2 = F.normalize(x2, p = 2, dim = 1)
            x = torch.cat((x0, x1, x2), 1) # (bs, 1810)
        else:
            x = torch.cat((x0.view(bs, -1), x1, x2), 1) # (bs, 1810)
        return x

    def get_embedding(self, x, rois):
        embed = self.roi_pooler(x, rois.view(-1, 5))
        embed = self.forward(embed)  # (bs, 28, 28) --> (bs, 50)

        return embed

class Agent(nn.Module):
    def __init__(self, dim=128, hidden=48, num_class=2):
        super(Agent, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden, num_class),
        )

        #init params
        #init_params(self.fc2)

    def forward(self, embed):
        logits = self.fc(embed)
        return logits

if __name__ == '__main__':
    import torch

    embed_net = EmbeddingNet()
    model = Agent(dim=1810, hidden=24, num_class=10)
    input = torch.rand(3,1,28,28)
    rois = torch.tensor([[0, 42, 42, 83, 83],
                         [1, 28, 28, 56, 56],
                         [2, 14, 14, 83, 83]], dtype=torch.float32) # 41, 29, 70
    output = embed_net(input)
    logits = model(output)
    print(output.shape)
    print(logits.shape, logits.max(1)[1])

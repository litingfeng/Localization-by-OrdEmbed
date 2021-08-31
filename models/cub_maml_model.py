"""
https://github.com/wyharveychen/CloserLookFewShot/blob/e03aca8a2d01c9b5861a5a816cd5d3fdfc47cd45/backbone.py
fast weight
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


class Conv2d_fw(nn.Conv2d):  # used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                        bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out

class Linear_fw(nn.Linear): #used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None #Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast) #weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear_fw, self).forward(x)
        return out

class Net_ae_proj(nn.Module):
    '''Autoencoder+ordinal'''
    def __init__(self, channel=1, pooling_size=None,
                 pooling_mode=None, init_weights: bool = True):
        super(Net_ae_proj, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channel, 10, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        )
        if pooling_mode == 'pool':
            self.RCNN_pooler = ROIPool((pooling_size, pooling_size), 1.0 / 2.)
        elif pooling_mode == 'align':
            self.RCNN_pooler = ROIAlign((pooling_size, pooling_size),
                                       1.0 / 2., 2)

        self.RCNN_top = nn.Sequential(
            nn.Linear(64 * pooling_size * pooling_size, 1024),
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

    def forward(self, x, rois):
        self.base_feat = self.encoder(x) # (bs, 64, 42, 42)
        pooled_feat = self.RCNN_pooler(self.base_feat, rois.view(-1, 5))
        top_feat = self.RCNN_top(pooled_feat.view(rois.shape[0], -1))
        return pooled_feat, top_feat

    def get_roi_embedding(self, rois):
        pooled_feat = self.RCNN_pooler(self.base_feat, rois.view(-1, 5))  # (bs, 64, 7, 7)
        top_feat = self.RCNN_top(pooled_feat.view(rois.shape[0], -1))
        return pooled_feat, top_feat

class Agent_ae(nn.Module):
    '''
    policy gradient model for the autoencoder embedding net defined in
    mnist_baseline_ae_model.py
    '''
    def __init__(self, rnn=1, poolsize=7, hidden_size=None, num_class=2):
        super(Agent_ae, self).__init__()
        self.conv = nn.Sequential(
            Conv2d_fw(64, 32, kernel_size=1, stride=1),
            nn.ReLU()
        )
        if rnn == 1:
            self.i2h = Linear_fw(32*poolsize*poolsize, hidden_size)
            self.h2h = Linear_fw(hidden_size, hidden_size)
            self.fc = Linear_fw(hidden_size, num_class)
        else:
            self.fc = nn.Sequential(
                Linear_fw(32*poolsize*poolsize, hidden_size),
                nn.ReLU(True),
                nn.Dropout(p=0.5),
                Linear_fw(hidden_size, num_class),
            )

    def forward(self, embed, h_t_prev=None):
        embed = self.conv(embed).view(embed.shape[0], -1) # (bs, 32*poolsize*poolsize)
        if h_t_prev is not None:
            h1 = self.i2h(embed)  # (bs, hidden_size)
            h2 = self.h2h(h_t_prev)  # (bs, hidden_size)
            h_t = F.relu(h1 + h2)  # produce h_t # (bs, hidden_size)
            logits = self.fc(h_t)
            draw = sample(F.softmax(logits, dim=1))
            return h_t, logits, draw
        else:
            logits = self.fc(embed)
            draw = sample(F.softmax(logits, dim=1))
            return logits, draw

if __name__ == '__main__':
    import torch
    from models.mnist_baseline_ae_model import EmbeddingNet
    net = Net_ae_proj(channel=3, pooling_mode='align', pooling_size=7)
    model = Agent_ae(rnn=1, poolsize=7, hidden_size=48, num_class=10)
    input = torch.rand(3,3,128,128)
    rois = torch.tensor([[0, 42, 42, 83, 83],
                         [1, 28, 28, 56, 56],
                         [2, 14, 14, 83, 83]], dtype=torch.float32) # 41, 29, 70
    output, top = net(input, rois)
    print('embed ', output.shape, ' top ', top.shape)
    h_t = torch.zeros(3,48, dtype=torch.float)
    h_t, logits, action = model(output, h_t)
    print('h_t ', h_t.shape,
          ' logits ', logits.shape,
          ' action ', action.shape)

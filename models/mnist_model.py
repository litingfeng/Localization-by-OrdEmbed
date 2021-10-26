# -*- coding: utf-8 -*-
# @Time : 9/8/21 1:17 PM
# @Author : Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.

import sys
sys.path.append('/research/cbim/vast/tl601/projects/faster-rcnn.pytorch/lib')
import torch.nn as nn
from model.roi_layers import ROIPool, ROIAlign


class AutoencoderProj(nn.Module):
    def __init__(self, channel=1, pooling_size=None, dim=512,
                 pooling_mode=None, init_weights: bool=True):
        super(AutoencoderProj, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(channel, 10, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 10, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(10, channel, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
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
            nn.Linear(1024, dim),
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

    def forward(self, x, rois=None):
        self.base_feat = self.encoder(x) # (bs, 64, 42, 42)
        #print('self.base_feat ', self.base_feat.shape)
        if rois is not None:
            pooled_feat = self.RCNN_pooler(self.base_feat, rois.view(-1, 5)).view(rois.shape[0], -1)
            #print('pooled_feat ', pooled_feat.shape)
            pooled_feat = self.RCNN_top(pooled_feat)
        x = self.decoder(self.base_feat)
        #print('x ', x.shape)
        if rois is not None:
            return x, pooled_feat
        return x

class AutoencoderProj_ag(AutoencoderProj):
    def __init__(self, channel=1, pooling_size=None, dim=512,
                 pooling_mode=None, init_weights: bool=True):
        super(AutoencoderProj_ag, self).__init__(channel=channel, pooling_size=pooling_size, dim=dim,
                                              pooling_mode=pooling_mode, init_weights=init_weights)

    def forward(self, x, rois):
        self.base_feat = self.encoder(x) # (bs, 64, 42, 42)
        pooled_feat = self.RCNN_pooler(self.base_feat, rois.view(-1, 5))
        top_feat = self.RCNN_top(pooled_feat.view(rois.shape[0], -1))
        return pooled_feat, top_feat

    def get_roi_embedding(self, rois):
        pooled_feat = self.RCNN_pooler(self.base_feat, rois.view(-1, 5))  # (bs, 64, 7, 7)
        top_feat = self.RCNN_top(pooled_feat.view(rois.shape[0], -1))
        return pooled_feat, top_feat

if __name__ == '__main__':
    import torch
    model = AutoencoderProj(channel=1, pooling_size=7,
                            pooling_mode='align')
    input = torch.rand(3,1,84,84)
    rois = torch.tensor([[0, 42, 42, 83, 83],
                         [1, 28, 28, 56, 56],
                         [2, 14, 14, 83, 83]], dtype=torch.float32)  # 41, 29, 70
    output, feat = model(input, rois)
    print('output ', output.shape, ' feat ', feat.shape)
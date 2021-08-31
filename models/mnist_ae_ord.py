"""
Created on 4/2/2021 12:33 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import sys
sys.path.append('/research/cbim/vast/tl601/projects/faster-rcnn.pytorch/lib')
import torch
import torch.nn as nn
from model.roi_layers import ROIPool, ROIAlign

class Autoencoder(nn.Module):
    def __init__(self, pooling_size=None,
                 pooling_mode=None, init_weights: bool = True):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3, stride=2, padding=1),
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
            nn.ConvTranspose2d(10, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        if pooling_mode == 'pool':
            self.RCNN_pooler = ROIPool((pooling_size, pooling_size), 1.0 / 2.)
        elif pooling_mode == 'align':
            self.RCNN_pooler = ROIAlign((pooling_size, pooling_size),
                                       1.0 / 2., 2)

        # self.RCNN_top = nn.Sequential(
        #     nn.Linear(64 * pooling_size * pooling_size, 1024),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(1024, 512),
        # )

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
        #print('x ', x.shape)
        if rois is not None:
            pooled_feat = self.RCNN_pooler(self.base_feat, rois.view(-1, 5)).view(x.shape[0], -1)
        #pooled_feat = self.RCNN_top(pooled_feat)
        x = self.decoder(self.base_feat)
        if rois is not None:
            return x, pooled_feat
        return x

    def get_embedding(self, x, rois):
        self.base_feat = self.encoder(x)  # (bs, 64, 42, 42)
        pooled_feat = self.RCNN_pooler(self.base_feat, rois.view(-1, 5))
        top_feat = self.RCNN_top(pooled_feat.view(rois.shape[0], -1))
        return pooled_feat, top_feat

class AutoencoderProj(nn.Module):
    def __init__(self, channel=1, pooling_size=None, dim=512,
                 pooling_mode=None, init_weights: bool = True):
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

    def get_embedding(self, x, rois):
        self.base_feat = self.encoder(x)  # (bs, 64, 42, 42)
        pooled_feat = self.RCNN_pooler(self.base_feat, rois.view(-1, 5))
        top_feat = self.RCNN_top(pooled_feat.view(rois.shape[0], -1))
        return pooled_feat, top_feat

class AutoencoderProj_cls(nn.Module):
    def __init__(self, channel=1, pooling_size=None, dim=512,
                 pooling_mode=None, init_weights: bool = True):
        super(AutoencoderProj_cls, self).__init__()
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

        self.Ord_top = nn.Sequential(
            nn.Linear(64 * pooling_size * pooling_size, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, dim),
        )
        self.Cls_top = nn.Sequential(
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
        if rois is not None:
            pooled_feat = self.RCNN_pooler(self.base_feat, rois.view(-1, 5)).view(rois.shape[0], -1)
            top_ord_feat = self.Ord_top(pooled_feat)
            top_cls_feat = self.Cls_top(pooled_feat)
        x = self.decoder(self.base_feat)
        #print('x ', x.shape)
        if rois is not None:
            return x, top_ord_feat, top_cls_feat
        return x

    def get_roi_embedding(self, rois):
        pooled_feat = self.RCNN_pooler(self.base_feat, rois.view(-1, 5)) # (bs, 64, 7, 7)
        top_ord_feat = self.Ord_top(pooled_feat.view(rois.shape[0], -1))
        top_cls_feat = self.Cls_top(pooled_feat.view(rois.shape[0], -1))
        return pooled_feat, top_ord_feat, top_cls_feat

if __name__ == '__main__':
    model = AutoencoderProj_cls(channel=1, pooling_size=7, pooling_mode='align')
    input = torch.rand(3,1,84,84)
    rois = torch.tensor([[0, 42, 42, 83, 83],
                         [1, 28, 28, 56, 56],
                         [2, 14, 14, 83, 83]], dtype=torch.float32)  # 41, 29, 70
    output, feat_ord, feat_cls = model(input, rois)
    print('output ', output.shape, ' feat_ord ', feat_ord.shape, ' feat_cls ', feat_cls.shape)
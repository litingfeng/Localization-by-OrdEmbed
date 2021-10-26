# -*- coding: utf-8 -*-
# @Time : 9/12/21 9:13 PM
# @Author : Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.

import sys
sys.path.append('/research/cbim/vast/tl601/projects/faster-rcnn.pytorch/lib')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.roi_layers import ROIPool, ROIAlign
from util.utils import sample

class Encoder_CUB(nn.Module):
    def __init__(self, pooling_size=None, pretrained=True, base='vgg16', fixed=3, dim=512,
                 pooling_mode=None, num_classes=10):
        super(Encoder_CUB, self).__init__()
        self.pooling_size = pooling_size
        self.pretrained = pretrained

        if base == 'vgg16':
            model_path = './models/vgg16_caffe.pth'
            vgg = models.vgg16()
            if self.pretrained:
                print("Loading pretrained weights from %s" % (model_path))
                state_dict = torch.load(model_path)
                vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})

            self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])
            # Fix the layers before conv3: 10, conv4: 17, conv5: 24, all: 30
            for layer in range(30):
                for p in self.RCNN_base[layer].parameters(): p.requires_grad = False
            # TODO deal with boxs smaller than poolsize*16
            self.RCNN_top = nn.Sequential(
                nn.Linear(512 * pooling_size * pooling_size, 1024),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(1024, dim)
            )
        elif base == 'res18':
            resnet = models.resnet18(pretrained=self.pretrained)
            # Build resnet.
            self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                           resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3)
            self.RCNN_top = nn.Sequential(resnet.layer4)
            # Fix blocks
            for p in self.RCNN_base[0].parameters(): p.requires_grad = False
            for p in self.RCNN_base[1].parameters(): p.requires_grad = False

            if fixed is not None:
                assert (0 <= fixed < 4)
                if fixed >= 3:
                    for p in self.RCNN_base[6].parameters(): p.requires_grad = False
                if fixed >= 2:
                    for p in self.RCNN_base[5].parameters(): p.requires_grad = False
                if fixed >= 1:
                    for p in self.RCNN_base[4].parameters(): p.requires_grad = False

        if pooling_mode == 'pool':
            self.RCNN_pooler = ROIPool((pooling_size, pooling_size), 1.0 / 16.)
        elif pooling_mode == 'align':
            self.RCNN_pooler = ROIAlign((pooling_size, pooling_size),
                                        1.0 / 16., 2)

        dout = 512 if base == 'vgg16' else 256
        # self.classifier = nn.Sequential(
        #     nn.Linear(dout * pooling_size * pooling_size, 1024),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(512, num_classes)
        # )
        if not pretrained:
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
        self.base_feat = self.RCNN_base(x) # vgg (bs, 512, 14, 14), res (bs,256,14,14)
        pooled_feat = self.RCNN_pooler(self.base_feat, rois.view(-1,5)) # (bs, 512, 7, 7)
        top_feat = self._head_to_tail(pooled_feat) # (bs, dim)
        #class_out = self.classifier(pooled_feat.view(pooled_feat.size(0), -1))
        return pooled_feat, top_feat#, class_out

    def get_roi_embedding(self, rois):
        pooled_feat = self.RCNN_pooler(self.base_feat, rois.view(-1, 5))  # (bs, 64, 7, 7)
        top_feat = self._head_to_tail(pooled_feat)  # (bs, 1024)
        return pooled_feat, top_feat

if __name__ == '__main__':
    model = Encoder_CUB(pooling_size=7, pretrained=True, dim=1024, pooling_mode='align')
    input = torch.rand(3,3,224,224)
    rois = torch.tensor([[0, 42, 42, 83, 83],
                         [1, 28, 28, 56, 56],
                         [2, 14, 14, 83, 83]], dtype=torch.float32) # 41, 29, 70
    pooled_feat, top_feat, class_out = model(input, rois)
    print('pooled_feat: ', pooled_feat.shape, '\ntop_feat: ', top_feat.shape, '\nclass_out ', class_out.shape)
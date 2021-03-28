"""
Created on 10/22/2020 12:26 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import torch
import torch.nn as nn
import torchvision.models as models

class VGG(nn.Module):

    def __init__(self, pretrained=True, num_classes=11):
        super(VGG, self).__init__()

        self.pretrained = pretrained
        model_path = '../vgg16_caffe.pth'
        vgg = models.vgg16()
        if self.pretrained:
            # use pretrained model
            print("Loading pretrained weights from %s" % (model_path))
            state_dict = torch.load(model_path)
            vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})

        vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

        self.features = nn.Sequential(*list(vgg.features._modules.values())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.top = vgg.classifier
        self.cls_score = nn.Linear(4096, num_classes)

    def forward(self, x):

        feature = self.features(x) # (bs, 512, 14, 14)
        x = self.avgpool(feature)
        x = torch.flatten(x, 1)
        x = self.top(x)
        x = self.cls_score(x)
        return x

if __name__ == '__main__':
    input = torch.rand(3,3,224,224)
    vgg = VGG(pretrained=True)
    output = vgg(input)

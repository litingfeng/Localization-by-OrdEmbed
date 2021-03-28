"""
Created on 3/10/2021 12:11 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import torch.nn as nn
from models.cumlink_model import LogisticCumulativeLink

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(1600, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 5)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

class Net(nn.Module):
    def __init__(self, embedding_net, n_classes, n_ang_classes,
                 mode, init_cutpoints='ordered'):
        super(Net, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.mode = mode
        self.nonlinear = nn.PReLU()
        self.fc = nn.Linear(5, n_classes)

        if self.mode == 'classify' or mode == 'cum_feat':
            angle_out = n_ang_classes if mode == 'classify' else 1
            self.angle_net = nn.Sequential(
                nn.Linear(5, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, angle_out, bias=False)
            )
        if self.mode == 'cum_feat':
            self.link = LogisticCumulativeLink(n_ang_classes,
                                          init_cutpoints=init_cutpoints)


    def forward(self, x):
        output = self.embedding_net(x)
        feature = self.nonlinear(output)
        scores = self.fc(feature)
        if self.mode == 'cum_feat' or self.mode == 'classify':
            output_o = self.angle_net(feature)
            if self.mode == 'classify':
                return feature, scores, output_o
            output_o = self.link(output_o)
            return feature, scores, output_o
        elif self.mode == 'cum_loss':
            return feature, scores

if __name__ == '__main__':
    import torch
    embedding_net = EmbeddingNet()
    net = Net(embedding_net, 2, 13, mode='classify')
    input = torch.rand(2,3,32,32)
    output = net(input)
    for o in output:
        print('o ', o.shape)

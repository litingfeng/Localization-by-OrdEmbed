"""
https://github.com/adambielski/siamese-triplet/blob/master/networks.py
Created on 5/23/2021 6:53 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingNet(nn.Module):
    def __init__(self, dim=256):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 1024),
                                nn.PReLU(),
                                nn.Linear(1024, dim),
                                # nn.PReLU(),
                                # nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


if __name__ == '__main__':
    import torch
    model = EmbeddingNet(dim=512)
    input = torch.rand(3,1,28,28)
    print('output ', model(input).shape)

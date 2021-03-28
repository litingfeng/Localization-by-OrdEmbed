"""
Created on 3/7/2021 3:11 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import torch.nn as nn
import torch.nn.functional as F
from util import sample

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

    def get_embedding(self, x):
        return self.forward(x)

class Net(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(Net, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(5, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        feature = self.nonlinear(output)
        scores = self.fc1(feature)
        return feature, scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))

class Agent(nn.Module):
    def __init__(self, dim=2, num_action=2, hidden_size=None):
        super(Agent, self).__init__()
        self.num_action = num_action
        dim_fc = dim if not hidden_size else hidden_size
        self.fc = nn.Sequential(
            nn.Linear(dim_fc, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_action)
        )
        if hidden_size:
            self.i2h = nn.Linear(dim, hidden_size)
            self.h2h = nn.Linear(hidden_size, hidden_size)

    def forward(self, feature, h_t_prev=None):
        #self.bn1 = nn.BatchNorm1d(2)
        # run through rnn
        if h_t_prev is not None:
            h1 = self.i2h(feature)  # (bs, hidden_size)
            h2 = self.h2h(h_t_prev)  # (bs, hidden_size)
            h_t = F.relu(h1 + h2)  # produce h_t # (bs, hidden_size)

            logits = self.fc(h_t)
            draw = sample(F.softmax(logits, dim=1))
            return h_t, logits, draw

        else:
            logits = self.fc(feature)
            draw = sample(F.softmax(logits, dim=1))

            return logits, draw
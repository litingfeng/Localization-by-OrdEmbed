"""
Created on 10/26/21 9:56 AM
@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import torch.nn as nn
import torch.nn.functional as F
from util.utils import sample

class Agent(nn.Module):
    def __init__(self, rnn=True, dim=64, poolsize=7, hidden_size=None, num_class=2):
        super(Agent, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, 32, kernel_size=1, stride=1),
            nn.ReLU()
        )
        if rnn:
            self.i2h = nn.Linear(32*poolsize*poolsize, hidden_size)
            self.h2h = nn.Linear(hidden_size, hidden_size)
            self.fc = nn.Linear(hidden_size, num_class)
        else:
            self.fc = nn.Sequential(
                nn.Linear(32*poolsize*poolsize, hidden_size),
                nn.ReLU(True),
                nn.Dropout(p=0.5),
                nn.Linear(hidden_size, num_class),
            )

    def forward(self, embed, h_t_prev=None):
        embed = self.conv(embed).view(embed.shape[0], -1) # (bs, 32*poolize*poolsize)
        if h_t_prev is not None:
            h1  = self.i2h(embed)    # (bs, hidden_size)
            h2  = self.h2h(h_t_prev) # (bs, hidden_size)
            h_t = F.relu(h1 + h2)
            logits = self.fc(h_t)
            draw = sample(F.softmax(logits, dim=1))
            return h_t, logits, draw
        else:
            logits = self.fc(embed)
            draw   = sample(F.softmax(logits, dim=1))
            return logits, draw


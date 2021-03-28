"""
https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/
c29ac479d33d5f2bc9a60b54b4822e60fc7819d0/RND%20Montezuma's%20revenge%20PyTorch/model.py
Created on 1/3/2021 12:25 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.

In RND there are 2 networks:
- Target Network: generates a constant output for a given state
- Prediction network: tries to predict the target network's output
"""
import torch
import torch.nn as nn
import numpy as np
from torch.nn import init

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class RNDModel(nn.Module):
    def __init__(self, input_size, device, output_size=None):
        super(RNDModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.device = device

        # Prediction network
        h_size = 50
        self.predictor = nn.Sequential(
            nn.Linear(self.input_size, h_size),
            #nn.BatchNorm1d(h_size),
            nn.ReLU(),
            # nn.Linear(h_size, h_size),
            # #nn.BatchNorm1d(h_size),
            # nn.ReLU(),
            # nn.Linear(h_size, 1),
            # nn.ReLU()
        )

        # Target network
        self.target = nn.Sequential(
            nn.Linear(self.input_size, h_size),
            #nn.BatchNorm1d(h_size),
            nn.ReLU(),
            # nn.Linear(h_size, h_size),
            # #nn.BatchNorm1d(h_size),
            # nn.ReLU(),
            # nn.Linear(h_size, 1),
            # nn.ReLU()
        )

        # Initialize the weights and biases
        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        # Set that target network is not trainable
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        #next_obs = torch.FloatTensor(next_obs).to(self.device)
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature

    # Calculate Intrinsic reward (prediction error)
    def compute_intrinsic_reward(self, next_obs):
        next_obs = torch.FloatTensor(next_obs).to(self.device)

        # Get target feature
        target_next_feature = self.target(next_obs)

        # Get prediction feature
        predict_next_feature = self.predictor(next_obs)

        # Calculate intrinsic reward TODO why / 2
        intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2

        return intrinsic_reward


if __name__ == '__main__':
    import torch
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from util.utils import RunningMeanStd

    input = torch.rand(2, 256) # input is state, in rnn, h_t?
    model = RNDModel(256, device).to(device)
    predict_feature, target_feature = model(input)
    print('predict_feature ', predict_feature.shape, ' target_feature ',
          target_feature.shape)
    intrinsic_loss = F.mse_loss(predict_feature, target_feature.detach(), reduction='none').mean(-1)
    print('intrinsic_loss ', intrinsic_loss.shape)

"""
Created on 2/25/2021 2:14 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import os, copy
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from actions import Actor
from util import cosine_simililarity
from torchvision import datasets, transforms
from RL import get_policy_loss

DEBUG = False
# dataloader class
class myCIFAR(datasets.CIFAR10):
    def __init__(self, root, subset, ratio=0.3, datapath=None, train=True, transform=None,
                 target_transform=None, download=True):
        super(myCIFAR, self).__init__(root, train=train, download=download,
                                      transform=transform, target_transform=target_transform)
        self.subset = subset
        self.ratio = ratio
        phase = 'train' if train else 'test'
        if not datapath:
            # select 4, 1
            self.newdata, self.newtargets = [], []
            for idx, digit in enumerate(self.targets):
                if digit == 4 or digit == 1:
                    self.newdata.append(idx)
                    target = 0 if digit == 4 else 1
                    self.newtargets.append(target)
            self.newdata, self.newtargets = np.array(self.newdata), \
                                        np.array(self.newtargets)
            self.rot_inds = np.random.choice(len(self.newdata),
                                             size=int(self.ratio*len(self.newdata)), replace=False) # select 100 to rotate
            print('rot number of 4 {}\nrot number of 1 {}'.format(
                len(np.where(self.newtargets[self.rot_inds] == 0)[0]),
                len(np.where(self.newtargets[self.rot_inds] == 1)[0])))
            self.norot_inds =np.array(list(set(range(len(self.newdata))) - set(self.rot_inds)))
            print('self.norot_inds ', self.norot_inds.shape)
            pickle.dump((self.newdata, self.newtargets, self.rot_inds, self.norot_inds),
                        open('{}_cifar_rot{}_more.pkl'.format(phase,
                                                              int(self.ratio*100)), 'wb'))
        else:
            self.newdata, self.newtargets, self.rot_inds, self.norot_inds = \
                pickle.load(open('{}_cifar_rot{}_more.pkl'.format(phase,
                                                                  int(self.ratio*100)), 'rb'))
        print('number of 4 {}\nnumber of 1 {}'.format(len(np.where(self.newtargets == 0)[0]),
                                                      len(np.where(self.newtargets == 1)[0])))
        print(' rot: {} norot_inds: {} '.format(self.rot_inds.shape, self.norot_inds.shape))

        # select which subset to train
        if self.subset == 'original_all': # use all original(non-roated) 200 samples
            self.data = [self.data[i] for i in self.newdata]
            self.targets = self.newtargets
            self.data = np.stack(self.data)
        elif self.subset == 'original_half': # use 100 non-rotated samples
            print('self.norot_inds ', self.norot_inds.shape)
            self.data = [self.data[self.newdata[i]] for i in self.norot_inds]
            self.targets = self.newtargets[self.norot_inds]
            self.data = np.stack(self.data)
        elif self.subset == 'half_half':
            #assert (len(self.rot_inds) == 3000)
            num_each = int(len(self.rot_inds) / 3.)
            self.orgdata = [self.data[i].transpose(2,0,1) for i in self.newdata] # HWC->CHW
            self.data = copy.deepcopy(self.orgdata)
            for i, inds in enumerate(self.rot_inds):
                k = i // num_each + 1
                self.data[inds] = np.rot90(
                    self.data[inds].copy(), k, axes=(-2,-1))
            self.data = np.stack(self.data).transpose(0, 2, 3, 1)
            self.targets = self.newtargets
            self.rot_dict = torch.zeros(len(self.data))
            self.rot_dict[self.rot_inds] = 1
        else:
            print('Not implementd')
            exit()

        print('subset [{}] data: {}'.format(self.subset, self.data.shape[0]))

        if self.subset == 'half_half':
            self.four_rot, self.one_rot = [], []
            for i in self.rot_inds:
                self.four_rot.append(i) if self.targets[i] == 0 else self.one_rot.append(i)
            self.four_norot, self.one_norot = [], []
            for i in self.norot_inds:
                self.four_norot.append(i) if self.targets[i] == 0 else self.one_norot.append(i)
            print('rot 4: {} rot 1: {}'.format(len(self.four_rot), len(self.one_rot)))
            print('nonrot 4: {} nonrot 1: {}'.format(len(self.four_norot), len(self.one_norot)))

    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]
        img = Image.fromarray(data)

        if DEBUG:
            for i in (10, 1200, 2200):
                inds = self.rot_inds[i]
                data, target = self.data[inds], self.targets[inds]
                img = Image.fromarray(data)
                f, ax = plt.subplots()
                ax.imshow(np.asarray(
                    Image.fromarray(self.orgdata[inds].transpose(1, 2, 0))), cmap='gray', interpolation='none')
                plt.title('org')
                print('target ', target)
                plt.show()

                f, ax = plt.subplots()
                ax.imshow(np.asarray(img), cmap='gray', interpolation='none')
                plt.title('trans')
                plt.show()
            exit()
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

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

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saveroot = os.path.join('/research/cbim/vast/tl601/results/selfpaced/rot50')

    dataset = myCIFAR(root='.', subset='half_half', ratio=0.5,
                      datapath='.',
                      train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]))
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=512, shuffle=True, num_workers=8)

    embedding_net = EmbeddingNet()
    net = Net(embedding_net, 2).to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    # load pretrained classifier
    ckpt = torch.load(os.path.join(saveroot, 'original_half', '14.pth.tar'))
    net.load_state_dict(ckpt['state_dict'])
    print('loaded ckpt epoch {} acc {:.2f}%'.format(ckpt['epoch'], ckpt['acc']))

    net.train()
    rewards, num_r_backs = [], []
    for i in range(4, 100):
        actor = Actor()
        for batch_idx, (data, target, index) in enumerate(train_loader):
            batch_size = data.shape[0]
            data, target = data.to(device), target.to(device)
            isrot = dataset.rot_dict[index].to(device)

            # forward classifier
            with torch.no_grad():
                feature, output = net(data)
                loss_org = criterion(output, target)
                output = F.softmax(output, dim=1)
                prob, pred = output.max(1, keepdim=True)

            # select actions
            less_conf = (isrot == 1).nonzero().flatten()
            #assert (len(less_conf) == 30)
            num_r_back = int(len(less_conf) * i / 100.)
            rot_inds = np.random.choice(len(less_conf), size=(num_r_back,), replace=False)
            actions = torch.ones_like(less_conf).view(-1, 1)
            actions[less_conf[rot_inds]] = 0

            # transform
            data[less_conf] = actor.takeaction(data[less_conf], actions)

            # compute reward
            tran_feature, tran_output = net(data)
            tran_output = F.softmax(tran_output, dim=1)
            trn_prob = torch.gather(tran_output[less_conf], 1, target[less_conf].unsqueeze(1))
            org_prob = torch.gather(output[less_conf], 1, target[less_conf].unsqueeze(1))
            reward = torch.sign(trn_prob - org_prob)

            rewards.append(reward.mean().item())
            num_r_backs.append(float(num_r_back) / len(less_conf) )

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(num_r_backs, rewards)
    ax.set_xlabel('% of rotate back')
    ax.set_ylabel('rewards')
    plt.show()



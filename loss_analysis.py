"""
Created on 2/25/2021 2:14 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import torch, os, copy
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
    print('target ', set(dataset.targets))
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=512, shuffle=True, num_workers=8)

    embedding_net = EmbeddingNet()
    net = Net(embedding_net, 2).to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    # load pretrained classifier
    for e in range(10):
        ckpt = torch.load(os.path.join(saveroot,
                                       'half_half', '{}.pth.tar'.format(e)))
        net.load_state_dict(ckpt['state_dict'])
        print('loaded ckpt epoch {} acc {:.2f}%'.format(ckpt['epoch'], ckpt['acc']))

        net.train()
        conf_loss, less_conf_loss = [], []

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
            conf = (isrot == 0).nonzero().flatten()
            conf_loss.append(loss_org[conf])
            less_conf_loss.append(loss_org[less_conf])

        conf_loss = torch.cat(conf_loss).cpu().numpy()
        less_conf_loss = torch.cat(less_conf_loss).cpu().numpy()
        loss_max = max(conf_loss.max(), less_conf_loss.max())
        #print('max ', loss_max)
        less_conf_loss = less_conf_loss / loss_max
        conf_loss = conf_loss / loss_max
        # print('conf ', conf_loss.shape, conf_loss[conf_loss>1])
        # print('less_conf_loss ', less_conf_loss.shape, less_conf_loss[less_conf_loss > 1])

        #fig = plt.figure()
        #plt.xlim(xmin=0, xmax=0.5)
        n, bins, patches = plt.hist(conf_loss, 100, density=True, facecolor='g', alpha=0.5)
        n, bins2, patches = plt.hist(less_conf_loss, 100, density=True, facecolor='r', alpha=0.5)
        #print('bins ', bins, '\nbins2 ', bins2)
        plt.title('epoch {} '.format(e))
        plt.show()



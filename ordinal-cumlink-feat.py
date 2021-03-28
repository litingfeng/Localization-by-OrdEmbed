"""
spacecutter on feature, jointly train with classifier
Created on 3/1/2021 8:29 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import os, wandb, argparse
import torch
import copy
import math
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
from PIL import Image
from models.cumlink_model import LogisticCumulativeLink, ascension_callback
import matplotlib.pyplot as plt
import numpy as np
from util import largest_rotated_rect
from torchvision import datasets, transforms
from losses.cumlink_losses import CumulativeLinkLoss

classes = {0.0: 0, 0.262: 1, 0.524: 2, 0.785: 3,
           1.047: 4, 1.309: 5, 1.571: 6, 1.833: 7,
           2.094: 8, 2.356: 9, 2.618: 10, 2.88: 11,
           3.142: 12}
tri = torch.tril(torch.ones(len(classes), len(classes)))
DEBUG = False
class MyRotationTransform:
    """Rotate by one of the given angles.
       rotation angle value in degrees, counter-clockwise.
    """
    def __init__(self, do_crop=False,
                 angles=np.array(range(0, 181, 15))):
        self.angles = angles
        self.do_crop = do_crop
        self.default_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def __call__(self, x):
        np.random.seed()
        angle1, angle2 = tuple(np.random.choice(self.angles, size=2,
                                          replace=False))
        angle1, angle2 = int(angle1), int(angle2)
        xi = TF.rotate(x, angle1)
        xj = TF.rotate(x, angle2)

        x0 = self.default_trans(x)
        xi = self.default_trans(xi)
        xj = self.default_trans(xj)

        return angle1, angle2, x0, xi, xj

class myCIFAR(datasets.CIFAR10):
    def __init__(self, root, datapath=None, subset='original_half',
                 train=True, transform=None,
                 target_transform=None, download=True):
        super(myCIFAR, self).__init__(root, train=train, download=download,
                                      transform=transform, target_transform=target_transform)
        phase = 'train' if train else 'test'
        self.subset = subset
        assert (self.subset == 'original_half')

        self.newdata, self.newtargets, self.rot_inds, self.norot_inds = \
            pickle.load(open('{}_cifar_rot{}_more.pkl'.format(phase,
                                                              int(0.5 * 100)), 'rb'))
        # if not datapath:
        #     # select 4, 1
        #     self.newdata, self.newtargets = [], []
        #     for idx, digit in enumerate(self.targets):
        #         if digit == 4 or digit == 1:
        #             self.newdata.append(idx)
        #             target = 0 if digit == 4 else 1
        #             self.newtargets.append(target)
        #     self.newdata, self.newtargets = np.array(self.newdata), \
        #                                     np.array(self.newtargets)
        #     pickle.dump((self.newdata, self.newtargets),
        #                 open('{}_oridinal.pkl'.format(phase), 'wb'))
        # else:
        #     self.newdata, self.newtargets = \
        #         pickle.load(open('{}_oridinal.pkl'.format(phase), 'rb'))
        print('number of 4 {}\nnumber of 1 {}'.format(len(np.where(self.newtargets == 0)[0]),
                                                      len(np.where(self.newtargets == 1)[0])))
        print(' rot: {} norot_inds: {} '.format(self.rot_inds.shape, self.norot_inds.shape))

        # self.data = [self.data[i] for i in self.newdata] # HWC
        # self.targets = self.newtargets
        # self.data = np.stack(self.data)
        if self.subset == 'original_half':  # use 1-ratio non-rotated samples
            self.data = [self.data[self.newdata[i]] for i in self.norot_inds]
            self.targets = self.newtargets[self.norot_inds]
            self.data = np.stack(self.data)

    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]
        img = Image.fromarray(data)

        if DEBUG:
            f, ax = plt.subplots()
            ax.imshow(np.asarray(img), cmap='gray', interpolation='none')
            plt.title('org')
            plt.show()

        img = self.transform(img)
        a1, a2, img0, img1, img2 = img
        if DEBUG:
            print('a1 ', a1, ' a2 ', a2)
            print('nexp ', img1.size, img2.size)
            f, ax = plt.subplots()
            ax.imshow(np.asarray(img1), cmap='gray', interpolation='none')
            plt.title('img1_nexp')
            plt.show()
            f, ax = plt.subplots()
            ax.imshow(np.asarray(img2), cmap='gray', interpolation='none')
            plt.title('img2_nexp')
            plt.show()
            exit()

        return math.radians(a1), math.radians(a2), \
            img0, img1, img2, target

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
        #print('output ', output.shape)
        output = self.fc(output)
        return output

class Net(nn.Module):
    def __init__(self, embedding_net, n_classes,
                 init_cutpoints: str = 'ordered'):
        super(Net, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(5, n_classes)
        self.fc_o = nn.Sequential(
                        nn.Linear(5, 64),
                        nn.ReLU(inplace=True),
                        nn.Linear(64, 128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, 1, bias=False)
                    )
        self.link = LogisticCumulativeLink(len(classes),
                                      init_cutpoints='ordered')

    def forward(self, x):
        output = self.embedding_net(x)
        feature = self.nonlinear(output)
        scores = self.fc1(feature)
        output_o = self.fc_o(feature)
        output_o = self.link(output_o)
        return feature, scores, output_o

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--savename', default='', type=str,
                        help='saving name')
    parser.add_argument('--step', default=50, type=int,
                        metavar='N', help='scheduler step')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--optimizer', type=str, default="Adam",
                        choices=['Adam', 'SGD'])
    parser.add_argument('-b', '--batch_size', default=512, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ['WANDB_NAME'] = args.savename
    wandb.init(project="selfpaced")
    saveroot = os.path.join('/research/cbim/vast/tl601/results/selfpaced/ordinal',
                            args.savename)
    os.makedirs(saveroot, exist_ok=True)

    train_set = myCIFAR(root='.',
                       datapath='.',
                      train=True, transform=MyRotationTransform())
    test_set = myCIFAR(root='.',
                        datapath='.',
                        train=False, transform=MyRotationTransform())
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=args.batch_size, shuffle=False, num_workers=8)

    embedding_net = EmbeddingNet()
    net = Net(embedding_net, 2).to(device)
    wandb.watch(net, log_freq=10)
    wandb.config.update(args)
    callback = ascension_callback()
    criterion = nn.CrossEntropyLoss(reduction='none')
    criterion_order = CumulativeLinkLoss()
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.1)

    '''
    training
    '''
    net.train()
    total_loss, total_l1, total_l2, total_lo = \
        [], [], [], []
    sub_as, total_subs = [], []
    accs, accs_o = [], []
    losses_ij, save_ts, feat_ijs = [], [], []
    for epoch in range(args.epochs):
        scheduler.step()
        train_losses, losses_i, losses_j, losses_order = \
            [], [], [], []
        sub_a, loss_sub = [], []
        save_t, loss_ijs, feat_ij = [], [], []
        correct_i, correct_j, correct_order = 0, 0, 0
        for batch_idx, (a1, a2, x0s, xis, xjs, target) in enumerate(train_loader):
            a1, a2, x0s, xis, xjs, target = a1.to(device), a2.to(device), \
                                            x0s.to(device), xis.to(device), xjs.to(device), \
                                            target.to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                feature_0, output_0, output_0_o = net(x0s)
            loss_0 = criterion(output_0, target)

            feature_i, output_i, output_i_o = net(xis)
            feature_j, output_j, output_j_o = net(xjs)
            loss_i = criterion(output_i, target)
            loss_j = criterion(output_j, target)

            # loss order
            a1 = torch.round(a1 * 10 ** 3) / 10 ** 3
            a1 = [classes[a1[i].item()] for i in range(a1.shape[0])]
            a1 = torch.from_numpy(np.array(a1))
            a2 = torch.round(a2 * 10 ** 3) / 10 ** 3
            a2 = [classes[a2[i].item()] for i in range(a2.shape[0])]
            a2 = torch.from_numpy(np.array(a2))
            a = torch.cat((a1, a2)).to(device)
            # no calibrate feature
            output_o = torch.cat((output_i_o, output_j_o))
            loss_order = criterion_order(output_o, a.unsqueeze(1))
            _, pred_order = output_o.max(1)
            correct_order += (pred_order == a).sum().item()

            output_i = F.softmax(output_i, dim=1)
            output_j = F.softmax(output_j, dim=1)
            _, pred_i = output_i.max(1, keepdim=True)
            _, pred_j = output_j.max(1, keepdim=True)
            correct_i += pred_i.eq(target.view_as(pred_i)).sum().item()
            correct_j += pred_j.eq(target.view_as(pred_j)).sum().item()

            #loss = loss_0.mean() + loss_order
            loss = loss_i.mean() + loss_j.mean() + loss_order
            loss.backward()
            optimizer.step()
            net.apply(callback)

            #scheduler.step()
            train_losses.append(loss.item())
            losses_i.append(loss_i.mean().item())
            losses_j.append(loss_j.mean().item())
            losses_order.append(loss_order.item())
            # sub_a.append(torch.mean(sub_angle).item())
            # loss_sub.append(torch.mean(loss_ij).item())
            loss_ijs.append(torch.cat((loss_0.unsqueeze(1),
                                       loss_i.unsqueeze(1), loss_j.unsqueeze(1)), 1).cpu())
            save_t.append(torch.cat((a1.unsqueeze(1), a2.unsqueeze(1)), 1).float())
            feat_ij.append(torch.cat((feature_0, feature_i, feature_j), 1).cpu())

        total_loss.append(np.mean(train_losses))
        total_l1.append(np.mean(losses_i))
        total_l2.append(np.mean(losses_j))
        total_lo.append(np.mean(losses_order))
        # sub_as.append(np.mean(sub_a))
        # total_subs.append(np.mean(loss_sub))
        losses_ij.append(loss_ijs)
        save_ts.append(save_t)
        feat_ijs.append(feat_ij)
        acc = 100. * correct_i / len(train_loader.dataset)
        acc_order = 100. * correct_order / (len(train_loader.dataset)*2.)
        accs.append(acc)
        accs_o.append(acc_order)
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'acc': acc
        }, os.path.join(saveroot, '{}.pth.tar'.format(epoch)))

        current_lr = scheduler.get_last_lr()[0]
        log_message = {"Total Loss": total_loss[-1],
                       "Loss i": total_l1[-1],
                       "Loss Order": total_lo[-1],
                       "Cls Acc": acc,
                       "Order Acc": acc_order,
                       'learning rate': current_lr}

        wandb.log(log_message, step=epoch)

    '''
    testing
    '''
    ckpt = torch.load(os.path.join(saveroot, '{}.pth.tar'.format(epoch)))
    net.load_state_dict(ckpt['state_dict'])
    print('loaded ckpt epoch {} acc {:.2f}%'.format(ckpt['epoch'], ckpt['acc']))
    net.eval()
    correct, correct_order = 0, 0
    with torch.no_grad():
        for batch_idx, (a1, a2, x0s, xis, xjs, target) in enumerate(test_loader):
            a1, a2, x0s, xis, xjs, target = a1.to(device), a2.to(device), \
                                            x0s.to(device), xis.to(device), xjs.to(device), \
                                            target.to(device)
            with torch.no_grad():
                feature_0, output_0, output_0_o = net(x0s)
            feature_i, output_i, output_i_o = net(xis)
            feature_j, output_j, output_j_o = net(xjs)
            loss_0 = criterion(output_0, target)
            loss_i = criterion(output_i, target)
            loss_j = criterion(output_j, target)


            a1 = torch.round(a1 * 10 ** 3) / 10 ** 3
            a1 = [classes[a1[i].item()] for i in range(a1.shape[0])]
            a1 = torch.from_numpy(np.array(a1))
            a2 = torch.round(a2 * 10 ** 3) / 10 ** 3
            a2 = [classes[a2[i].item()] for i in range(a2.shape[0])]
            a2 = torch.from_numpy(np.array(a2))
            a = torch.cat((a1, a2)).to(device)

            output_o = torch.cat((output_i_o, output_j_o))
            _, pred_order = output_o.max(1)
            correct_order += (pred_order == a).sum().item()

            output = F.softmax(output_0, dim=1)
            prob, pred = output.max(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    acc = 100. * correct / len(test_loader.dataset)
    acc_order = 100. * correct_order / (len(test_loader.dataset)*2.0)

    wandb.run.summary.update({'Test acc': acc,
               'Test acc_o': acc_order})

    print('acc in eval mode: {:.2f}% acc_order: {:.2f}'.format(acc, acc_order))
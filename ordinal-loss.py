"""
non-parametic ordinal loss on classifier loss
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
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

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
        # Center crop to ommit black noise on the edges
        # if self.do_crop == True: # expand+crop = no expand
        #     xi_w, xi_h = xi.size
        #     lrr_width, lrr_height = largest_rotated_rect(xi_w, xi_h,
        #                                                  angle1)
        #     print('img1 lrr_width, lrr_height ', lrr_width, lrr_height)
        #     xi_crop = TF.center_crop(xi, [lrr_height, lrr_width])
        #     xj_w, xj_h = xj.size
        #     lrr_width, lrr_height = largest_rotated_rect(xj_w, xj_h,
        #                                                  angle2)
        #     print('img2 lrr_width, lrr_height ', lrr_width, lrr_height)
        #     xj_crop = TF.center_crop(xj, [lrr_height, lrr_width])
        #     # image = tf.image.resize_images(resized_image, [output_height, output_width],
        #     #                                method=tf.image.ResizeMethod.BILINEAR, align_corners=False)

        return angle1, angle2, x0, xi, xj

class myCIFAR(datasets.CIFAR10):
    def __init__(self, root, datapath=None, train=True, transform=None,
                 target_transform=None, download=True):
        super(myCIFAR, self).__init__(root, train=train, download=download,
                                      transform=transform, target_transform=target_transform)
        phase = 'train' if train else 'test'

        if not datapath:
            assert (1==0)
            # select 4, 1
            self.newdata, self.newtargets = [], []
            for idx, digit in enumerate(self.targets):
                if digit == 4 or digit == 1:
                    self.newdata.append(idx)
                    target = 0 if digit == 4 else 1
                    self.newtargets.append(target)
            self.newdata, self.newtargets = np.array(self.newdata), \
                                            np.array(self.newtargets)
            pickle.dump((self.newdata, self.newtargets),
                        open('{}_oridinal.pkl'.format(phase), 'wb'))
        else:
            # self.newdata, self.newtargets = \
            #     pickle.load(open('{}_oridinal.pkl'.format(phase), 'rb'))
            self.newdata, self.newtargets, self.rot_inds, self.norot_inds = \
                pickle.load(open('{}_cifar_rot{}_more.pkl'.format(phase,
                                                                  int(0.5 * 100)), 'rb'))

        # self.data = [self.data[i] for i in self.newdata] # HWC
        # self.targets = self.newtargets
        # self.data = np.stack(self.data)
        if True:  # use 1-ratio non-rotated samples
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

def feed_data(model, data_loader, is_train):
    if is_train:
        phase = 'Train'
        model.train()
    else:
        phase = 'Test'
        model.eval()

    train_losses, losses_0, losses_i, losses_order = \
        [], [], [], []
    correct_0, correct_i, correct_order, correct_order_sub = 0, 0, 0, 0
    len_dataloader = len(data_loader)
    for batch_idx, (ai, aj, x0s,
                    xis, xjs, target) in enumerate(data_loader):
        batch_size = x0s.shape[0]
        ai, aj, x0s, xis, xjs, target = ai.to(device), aj.to(device), \
                                        x0s.to(device), xis.to(device), \
                                        xjs.to(device), target.to(device)

        optimizer.zero_grad()
        p = float(batch_idx + epoch * len_dataloader) / args.epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        feature_0, output_0 = net(x0s)
        feature_i, output_i = net(xis)
        feature_j, output_j = net(xjs)
        loss_0 = criterion(output_0, target).view(-1,1)
        loss_i = criterion(output_i, target).view(-1,1)
        loss_j = criterion(output_j, target).view(-1,1)

        # compute loss order
        a = torch.stack((ai, aj), 1)  # (bs, 2)
        l = torch.stack((loss_i, loss_j), 1)  # (bs, 2, 1)

        inds = torch.min(a, 1).indices
        p = l[torch.arange(0, batch_size), inds, :]
        n = l[torch.arange(0, batch_size), 1 - inds, :]
        loss_order = triplet_loss(loss_0, p, n)

        # update model
        loss = alpha * loss_0.mean() + (1-alpha) * loss_order * args.lamb
        # loss = (1-alpha)* loss_0.mean() + alpha * loss_order * args.lamb
        if phase == 'Train':
            loss.backward()
            optimizer.step()

        # compute accuracy
        output_i = F.softmax(output_i, dim=1)
        output_0 = F.softmax(output_0, dim=1)
        _, pred_i = output_i.max(1, keepdim=True)
        _, pred_0 = output_0.max(1, keepdim=True)
        correct_i += pred_i.eq(target.view_as(pred_i)).sum().item()
        correct_0 += pred_0.eq(target.view_as(pred_0)).sum().item()

        loss_ij = loss_i.view(-1) - loss_j.view(-1)
        sub_angle = ai - aj
        mask = (torch.sign(loss_ij) == torch.sign(sub_angle))
        correct_order_sub += mask.sum().item()

        d_a0a1 = F.pairwise_distance(loss_0, loss_i)
        d_a0a2 = F.pairwise_distance(loss_0, loss_j)
        d = torch.stack((d_a0a1, d_a0a2), 1)  # (bs, 2)
        inds_d = torch.min(d, 1).indices
        correct_order += (inds == inds_d).sum().item()

        train_losses.append(loss.item())
        losses_0.append(loss_0.mean().item())
        losses_i.append(loss_i.mean().item())
        losses_order.append(loss_order.item())
        if batch_idx % args.log_interval == 0:
            if batch_idx == 0:
                print('\n')
            print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                phase, epoch, batch_idx * batch_size,
                len(data_loader.dataset),
                100. * batch_idx / len_dataloader, np.mean(train_losses)))

    total_loss = np.mean(train_losses)
    acc = 100. * correct_0 / len(data_loader.dataset)
    acc_o = 100. * correct_order / len(data_loader.dataset)
    acc_sub = 100. * correct_order_sub / len(data_loader.dataset)

    log_message = {"{} Total Loss".format(phase): total_loss,
                   "{} Cls Loss_i".format(phase): np.mean(losses_i),
                   "{} Cls Loss_0".format(phase): np.mean(losses_0),
                   "{} Order Loss".format(phase): np.mean(losses_order),
                   '{} Cls Acc'.format(phase): acc,
                   '{} Ord Acc'.format(phase): acc_o,
                   '{} Ord Acc_sub'.format(phase): acc_sub,
                   }
    if phase == 'Train':
        current_lr = scheduler.get_last_lr()[0]
        log_message.update({'learning rate': current_lr})

    wandb.log(log_message, step=epoch)

    return total_loss, acc_sub

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--savename', default='', type=str,
                        help='saving name')
    parser.add_argument('--step', default=50, type=int,
                        metavar='N', help='scheduler step')
    parser.add_argument('--norm', default=0, type=int,
                        help='if 1 then norm feature')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--optimizer', type=str, default="Adam",
                        choices=['Adam', 'SGD'])
    parser.add_argument('--margin', default=0., type=float,
                        help='margin of loss')
    parser.add_argument('-b', '--batch_size', default=512, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lamb', default=1.0, type=float,
                        help='weight of loss order')
    parser.add_argument('--log_interval', default=10, type=int,
                        metavar='N', help='print frequency')
    parser.add_argument('--patience', default=50, type=int,
                        help='number of epochs to tolerate the no improvement of val acc')

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

    criterion = nn.CrossEntropyLoss(reduction='none')
    triplet_loss = nn.TripletMarginLoss(margin=args.margin, p=2)
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.1)

    best_acc = 0.
    save_model = False
    no_improve_epoch = 0
    for epoch in range(args.epochs):
        train_log = feed_data(net, data_loader=train_loader, is_train=True)
        scheduler.step()

        with torch.no_grad():
            test_loss, test_acc = feed_data(net, data_loader=test_loader, is_train=False)

            # save model
            if test_acc > best_acc:
                save_model = True
                best_acc = test_acc
                no_improve_epoch = 0
            else:
                save_model = False
                no_improve_epoch += 1
            if save_model:  # TODO
                torch.save({
                    'epoch': epoch,
                    'state_dict': net.state_dict(),
                    'acc': test_acc
                }, os.path.join(saveroot, 'best.pth.tar'))
                pass

            # break training
            if no_improve_epoch > args.patience:
                print('stop training...')
                break

        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'acc': test_acc
        }, os.path.join(saveroot, '{}.pth.tar'.format(epoch)))

"""
order regression angles

Created on 3/10/2021 12:03 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import os, wandb
import torch, copy

import torch.nn as nn
import torch.nn.functional as F
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from actions2 import Actor
from util import cosine_simililarity, sample
from torchvision import datasets, transforms
from RL import get_policy_loss
from opts import parser
from datasets.cifar_regression import myCIFAR
from transform import MyRotationTransform
from models.regression_model import Net, EmbeddingNet
from models.cumlink_model import ascension_callback, LogisticCumulativeLink
from losses.cumlink_losses import CumulativeLinkLoss

DEBUG = True
def init_dataloader():
    train_set = myCIFAR(root='.', datapath='.', train=True,
                        transform=MyRotationTransform(angles=np.array(range(0, 181, args.angle_step)),
                                                      num=3))

    test_set = myCIFAR(root='.', datapath='.',
                       train=False,
                       transform=MyRotationTransform(angles=np.array(range(0, 181, args.angle_step)),
                                                     num=3))
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size,
                                               shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=args.batch_size,
                                              shuffle=False, num_workers=8)
    print('total images of 4 & 1: ', len(train_set), ' test: ', len(test_set))

    return train_loader, test_loader

def init_model(n_a_classes):
    embedding_net = EmbeddingNet()
    net = Net(embedding_net, 2, n_a_classes, mode=args.mode).to(device)

    return net

def feed_data(model, data_loader, is_train):
    if is_train:
        phase = 'Train'
        model.train()
    else:
        phase = 'Test'
        model.eval()

    len_dataloader = len(data_loader)
    correct_cls, correct_ang = 0, 0
    total_loss, losses_0, losses_i, losses_order = [], [], [], []
    for batch_idx, (ai, aj, x0s,
                    xis, xjs, target) in enumerate(data_loader):
        batch_size = x0s.shape[0]
        ai, aj, x0s, xis, xjs, target = ai.to(device), aj.to(device), \
                                        x0s.to(device), xis.to(device), \
                                        xjs.to(device), target.to(device)
        optimizer.zero_grad()

        outputs_0 = net(x0s)
        outputs_i = net(xis)
        outputs_j = net(xjs)
        loss_0 = criterion(outputs_0[1], target)
        loss_i = criterion(outputs_i[1], target)
        loss_j = criterion(outputs_j[1], target)

        # compute loss
        a = torch.cat((ai, aj)) # (2*bs,)
        if args.mode == 'cum_feat':
            output_o = torch.cat((outputs_i[2], outputs_j[2])) # (2*bs. n_a_classes)
            loss_order = criterion_order(output_o, a.unsqueeze(1))
        elif args.mode == 'cum_loss':
            loss_ij = torch.cat((loss_i, loss_j)) # (2*bs,)
            output_o = link(loss_ij.unsqueeze(1)) # (2*bs. n_a_classes)
            loss_order = criterion_order(output_o, a.unsqueeze(1))
        elif args.mode == 'classify':
            output_o = torch.cat((outputs_i[2], outputs_j[2]))  # (2*bs. n_a_classes)
            loss_order = criterion_order(output_o, a)
        else:
            print('Not implemented mode')
            exit()

        loss = loss_0.mean() + loss_order

        if phase == 'Train':
            loss.backward()
            optimizer.step()
            if args.mode == 'cum_feat':
                net.apply(callback)
            elif args.mode == 'cum_loss':
                link.apply(callback)

        # compute accuracy
        output_0 = F.softmax(outputs_0[1], dim=1)
        _, pred_0 = output_0.max(1, keepdim=True)
        correct_cls += pred_0.eq(target.view_as(pred_0)).sum().item()

        _, pred_order = output_o.max(1)
        correct_ang += (pred_order == a).sum().item()

        total_loss.append(loss.item())
        losses_0.append(loss_0.mean().item())
        losses_i.append(loss_i.mean().item())
        losses_order.append(loss_order.item())
        if batch_idx % args.log_interval == 0:
            if batch_idx == 0:
                print('\n')
            print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                phase, epoch, batch_idx * batch_size,
                len(data_loader.dataset),
                100. * batch_idx / len_dataloader, np.mean(total_loss)))

    acc = 100. * correct_cls / len(data_loader.dataset)
    acc_o = 100. * correct_ang / (len(data_loader.dataset)*2.)

    log_message = {"{} Total Loss".format(phase): np.mean(total_loss),
                   "{} Cls Loss_i".format(phase): np.mean(losses_i),
                   "{} Cls Loss_0".format(phase): np.mean(losses_0),
                   "{} Order Loss".format(phase): np.mean(losses_order),
                   '{} Cls Acc'.format(phase): acc,
                   '{} Angle Acc'.format(phase): acc_o,
                   }
    if phase == 'Train':
        current_lr = scheduler.get_last_lr()[0]
        log_message.update({'learning rate': current_lr})
    wandb.log(log_message, step=epoch)

    return total_loss, acc_o

if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ['WANDB_NAME'] = args.savename
    wandb.init(project="selfpaced")
    saveroot = os.path.join('/research/cbim/vast/tl601/results/selfpaced/ordinal',
                            args.savename)
    os.makedirs(saveroot, exist_ok=True)

    train_loader, test_loader = init_dataloader()

    net = init_model(n_a_classes=len(train_loader.dataset.angles))
    if args.mode == 'cum_loss':
        link = LogisticCumulativeLink(len(train_loader.dataset.angles),
                                      init_cutpoints='ordered').to(device)
    wandb.watch(net, log_freq=10)
    wandb.config.update(args)

    criterion = nn.CrossEntropyLoss(reduction='none')
    if args.mode == 'cum_loss' or args.mode == 'cum_feat':
        criterion_order = CumulativeLinkLoss()
    else:
        criterion_order = nn.CrossEntropyLoss()
    callback = ascension_callback()

    if args.mode == 'cum_loss':
        params = list(net.parameters())+list(link.parameters())
    else:
        params = net.parameters()
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=args.step, gamma=0.1)

    best_acc = 0.
    save_model = False
    no_improve_epoch = 0
    for epoch in range(args.epochs):
        train_log = feed_data(net, data_loader=train_loader, is_train=True)
        scheduler.step()

        with torch.no_grad():
            test_loss, test_acc = feed_data(net, data_loader=test_loader,
                                            is_train=False)

            # save model
            if test_acc > best_acc:
                save_model = True
                best_acc = test_acc
                no_improve_epoch = 0
            else:
                save_model = False
                no_improve_epoch += 1
            if save_model:
                torch.save({
                    'epoch': epoch,
                    'state_dict': net.state_dict(),
                    'acc': test_acc
                }, os.path.join(saveroot, 'best.pth.tar'))

            # break training
            if no_improve_epoch > args.patience:
                print('stop training...')
                break

    torch.save({
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'acc': test_acc
    }, os.path.join(saveroot, 'last.pth.tar'))
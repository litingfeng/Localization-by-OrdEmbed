"""
learn ordinal embedding
Created on 3/1/2021 8:29 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import os, wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from transform import MyBoxTransform
from opts import parser
from util.augmentations import Compose
from util.data_aug import Resize
import numpy as np
from models.mnist_model import EmbeddingNet, Net
from datasets.clutter_mnist import MNIST_CoLoc

DEBUG = False

def init_dataloader():
    kwargs = {'num_workers': 8, 'pin_memory': True}

    train_transform = Compose([Resize(84)])
    test_transform = Compose([Resize(84)])
    trainset = MNIST_CoLoc(root='.', train=True, digit=4,
                           datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                           clutter=1, transform=train_transform,
                           target_transform=MyBoxTransform())
    testset = MNIST_CoLoc(root='.', train=False, digit=4,
                          datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                          clutter=1, transform=test_transform,
                          target_transform=MyBoxTransform())

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    print('total train image: ', len(train_loader.dataset), ' test image: ',
          len(test_loader.dataset))
    return train_loader, test_loader

def init_model():
    embedding_net = EmbeddingNet(norm=args.norm)
    net = Net(embedding_net, pooling_size=args.pooling_size,
              pooling_mode=args.pooling_mode).to(device)

    return net

def feed_data(model, data_loader, is_train):
    if is_train:
        phase = 'Train'
        model.train()
    else:
        phase = 'Test'
        model.eval()

    train_losses = []
    correct, correct_sub = 0, 0
    len_dataloader = len(data_loader)
    for batch_idx, (x0s, target, ti, tj, pixi, pixj) in enumerate(data_loader):
        batch_size = x0s.shape[0]
        x0s, target, ti, tj, pixi, pixj = x0s.to(device), \
                              target.to(device).float(), \
                              ti.to(device), tj.to(device), \
                              pixi.to(device), pixj.to(device)

        optimizer.zero_grad()

        # rois
        rois_0 = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                          target.to(device)), dim=1)
        rois_i = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                            ti.to(device)), dim=1)
        rois_j = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                            tj.to(device)), dim=1)

        feature_0 = net(x0s, rois_0) # (bs, 1810)
        feature_i = net(x0s, rois_i)
        feature_j = net(x0s, rois_j)

        # compute loss order
        pix = torch.stack((pixi, pixj), 1)  # (bs, 2)
        f = torch.stack((feature_i, feature_j)).permute(1, 0, 2)  # (bs, 2, dim)

        inds = torch.min(pix, 1).indices
        p = f[torch.arange(0, batch_size), inds, :]
        n = f[torch.arange(0, batch_size), 1 - inds, :]
        proto = torch.mean(feature_0, dim=0).view(1,-1).repeat(batch_size,1)
        loss = triplet_loss(proto, p, n)
        #loss = triplet_loss(feature_0, p, n)

        # update model
        if phase == 'Train':
            loss.backward()
            optimizer.step()

        # compute accuracy
        # d_a0a1 = F.pairwise_distance(feature_0, feature_i)
        # d_a0a2 = F.pairwise_distance(feature_0, feature_j)
        d_a0a1 = F.pairwise_distance(proto, feature_i)
        d_a0a2 = F.pairwise_distance(proto, feature_j)
        d = torch.stack((d_a0a1, d_a0a2), 1)  # (bs, 2)
        inds_d = torch.min(d, 1).indices
        correct += (inds == inds_d).sum().item()

        sub_d = d_a0a1 - d_a0a2
        sub_pix = pixi - pixj
        mask = (torch.sign(sub_d) == torch.sign(sub_pix))
        correct_sub += mask.sum().item()

        train_losses.append(loss.item())
        if batch_idx % args.log_interval == 0:
            if batch_idx == 0:
                print('\n')
            print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                phase, epoch, batch_idx * batch_size,
                len(data_loader.dataset),
                100. * batch_idx / len_dataloader, np.mean(train_losses)))

    total_loss = np.mean(train_losses)
    acc = 100. * correct / len(data_loader.dataset)
    acc_sub = 100. * correct_sub / len(data_loader.dataset)

    log_message = {"{} Total Loss".format(phase): total_loss,
                   '{} Ord Acc'.format(phase): acc,
                   '{} Ord Acc_sub'.format(phase): acc_sub,
                   }
    if phase == 'Train':
        current_lr = scheduler.get_last_lr()[0]
        log_message.update({'learning rate': current_lr})

    wandb.log(log_message, step=epoch)

    return total_loss, acc_sub

if __name__ == "__main__":

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ['WANDB_NAME'] = args.savename
    wandb.init(project="selfpaced")
    saveroot = os.path.join('/research/cbim/vast/tl601/results/selfpaced/ordinal',
                            args.savename)
    os.makedirs(saveroot, exist_ok=True)

    train_loader, test_loader = init_dataloader()

    net = init_model()
    wandb.watch(net, log_freq=10)
    wandb.config.update(args)

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
            if save_model:
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
    }, os.path.join(saveroot, 'last.pth.tar'))

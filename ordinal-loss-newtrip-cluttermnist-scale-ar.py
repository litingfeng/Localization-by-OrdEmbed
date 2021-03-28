"""
learn ordinal embedding with scale
Created on 3/1/2021 8:29 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import os, wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from opts import parser
from util.augmentations import Compose
from util.data_aug import Resize
import numpy as np
from util.utils import generate_boxes
from models.mnist_scale_model import Net
from datasets.clutter_mnist_scale_anchor import MNIST_CoLoc

DEBUG = False

def init_dataloader():
    kwargs = {'num_workers': 8, 'pin_memory': True}

    train_transform = Compose([Resize(84)])
    test_transform = Compose([Resize(84)])
    # anchors = generate_boxes(base_size=16, ratios=[0.5, 1.0, 2.0],
    #                          min_box_side=21,
    #                          scales=np.array([1.3, 2.0, 2.5, 2.0, 3.5, 4.0, 4.5]))
    # for larger digits, and more anchors
    anchors = generate_boxes(base_size=4,
                             feat_height=21, feat_width=21, img_size=84,
                             feat_stride=4,
                             ratios=np.linspace(0.5, 3.0, num=10),
                             min_box_side=28,
                             scales=np.array(range(7, 20)))
    print('number of anchors for 84*84 image ', anchors.shape[0])
    trainset = MNIST_CoLoc(root='.', train=True, digit=args.digit,
                           datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                           clutter=1, anchors= anchors, transform=train_transform)
    testset = MNIST_CoLoc(root='.', train=False, digit=args.digit,
                          datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                          clutter=1, anchors=anchors, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    print('total train image: ', len(train_loader.dataset), ' test image: ',
          len(test_loader.dataset))
    return train_loader, test_loader

def init_model():
    net = Net(pooling_size=args.pooling_size,
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
    for batch_idx, (x0s, target, ti, tj, ioui, iouj) in enumerate(data_loader):
        batch_size = x0s.shape[0]
        x0s, target, ti, tj, ioui, iouj = x0s.to(device), \
                              target.to(device).float(), \
                              ti.to(device), tj.to(device), \
                              ioui.to(device), iouj.to(device)

        optimizer.zero_grad()

        # rois
        rois_0 = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                          target.to(device)), dim=1)
        rois_i = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                            ti.to(device)), dim=1)
        rois_j = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                            tj.to(device)), dim=1)

        feature_0 = net(x0s, rois_0) # (bs, 1024)
        feature_i = net(x0s, rois_i)
        feature_j = net(x0s, rois_j)

        # compute loss order
        iou = torch.stack((ioui, iouj), 1)  # (bs, 2)
        f = torch.stack((feature_i, feature_j)).permute(1, 0, 2)  # (bs, 2, dim)

        inds = torch.max(iou, 1).indices
        p = f[torch.arange(0, batch_size), inds, :]
        n = f[torch.arange(0, batch_size), 1 - inds, :]
        proto = torch.mean(feature_0, dim=0).view(1,-1).repeat(batch_size,1)
        #loss = triplet_loss(proto, p, n)
        loss = triplet_loss(feature_0, p, n)

        # update model
        if phase == 'Train':
            loss.backward()
            optimizer.step()

        # compute accuracy
        d_a0a1 = F.pairwise_distance(feature_0, feature_i)
        d_a0a2 = F.pairwise_distance(feature_0, feature_j)
        sub_pix = iouj - ioui
        sub_d = d_a0a1 - d_a0a2
        mask = (torch.sign(sub_d) == torch.sign(sub_pix))
        correct += mask.sum().item()

        d_a0a1 = F.pairwise_distance(proto, feature_i)
        d_a0a2 = F.pairwise_distance(proto, feature_j)
        sub_d = d_a0a1 - d_a0a2
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
                   '{} Ord Acc Inst'.format(phase): acc,
                   '{} Ord Acc Proto'.format(phase): acc_sub,
                   }
    if phase == 'Train':
        current_lr = scheduler.get_last_lr()[0]
        log_message.update({'learning rate': current_lr})

    wandb.log(log_message, step=epoch)

    return total_loss, acc

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

    if args.evaluate == 1:
        # agent = Agent(dim=1024, num_class=10,
        #               hidden_size=args.hidden_size).to(device)
        # model_path = os.path.join(saveroot, 'coloc_rl_seq10_scale_ent5_joint',
        #                           'best.pth.tar')
        model_path = os.path.join(saveroot, 'best.pth.tar')
        print('loading ckpt from ', model_path)
        ckpt = torch.load(model_path)
        state_dict = ckpt['state_dict']
        epoch = ckpt['epoch']
        best_loss = ckpt['acc']
        net.load_state_dict(state_dict)
        print(("=> loaded checkpoint epoch {} {}".format(epoch, best_loss)))
        train_loss, train_acc = feed_data(net, data_loader=train_loader, is_train=False)
        test_loss, test_acc = feed_data(net, data_loader=test_loader, is_train=False)
        print('train_acc ', train_acc, ' test_acc ', test_acc)
        exit()


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

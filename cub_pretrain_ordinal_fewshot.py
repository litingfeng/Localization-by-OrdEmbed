"""
pretrain ordinal embedding for one class setting
https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
Created on 5/3/2021 2:54 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import os
from opts import parser
import wandb
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.optim as optim
from util.data_aug import *
from util.bbox_util import *
from util import joint_transforms as t
from util.utils import generate_boxes
from datasets.cub_fewshot_loose import CUB_fewshot, init_sampler
from models.cub_scale_model import Net
import matplotlib.pyplot as plt
from util.utils import convert_image_np
from constants import IMAGENET_MEAN, IMAGENET_STD

DEBUG = False

def init_dataloader():
    anchors = generate_boxes(base_size=16, feat_height=14, feat_width=14,
                             min_box_side=25, img_size=args.img_size,
                             feat_stride=16, ratios=np.linspace(0.3, 3.5, num=15),
                             scales=np.array(range(2, 13)))
    # anchors = generate_boxes(base_size=4,
    #                          feat_height=21, feat_width=21, img_size=84,
    #                          feat_stride=4,
    #                          ratios=np.linspace(0.3, 3.5, num=15),
    #                          min_box_side=16,
    #                          scales=np.array(range(4, 20)))
    transform = t.Compose([
                              t.ConvertFromPIL(),
                              t.ToPercentCoords(),
                              t.Resize(args.img_size),
                              t.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                              t.ToTensor()  # no change to (0, 1)
                          ])
    trainset = CUB_fewshot('/research/cbim/vast/tl601/Dataset/CUB_200_2011',
                          anchors=anchors, mode='base', img_size=args.img_size,
                          loose=args.loose,
                          transform=transform)
    testset = CUB_fewshot('/research/cbim/vast/tl601/Dataset/CUB_200_2011',
                          anchors=anchors, mode='val',img_size=args.img_size,
                          loose=args.loose,
                          transform=transform)

    train_batchsampler = init_sampler(args.classes_per_set, args.samples_per_class,
                                    trainset.meta['image_labels'],
                                      args.iterations, 'train')
    test_batchsampler = init_sampler(args.classes_per_set, args.samples_per_class,
                                     testset.meta['image_labels'],
                                     args.iterations_test, 'val')
    kwargs = {'num_workers': 8, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        #batch_sampler=train_batchsampler)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, **kwargs)
        #batch_sampler=test_batchsampler)
    return train_loader, test_loader

def feed_data(model, data_loader, is_train, upload=True):
    if is_train:
        phase = 'Train'
        model.train()
    else:
        phase = 'Test'
        model.eval()

    total_losses, recon_losses, ord_losses = [], [], []
    correct_inst, correct_proto, correct_inst_other, correct_proto_other = 0, 0, 0, 0
    len_dataloader = len(data_loader)
    count = 0
    for batch_idx, (data, target, ti, tj, ioui, iouj) in enumerate(data_loader):
        batch_size = data.shape[0]
        # if batch_size < args.samples_per_class:
        #     continue
        data, target, ti, tj, ioui, iouj = data.to(device), target.to(device).float(), \
                                           ti.to(device), tj.to(device), \
                                           ioui.to(device), iouj.to(device)

        if phase == 'Train':
            optimizer.zero_grad()

        '''split to support set and target set '''
        num = batch_size % args.samples_per_class
        if num != 0:
            if num == 1: continue
            data, target, ti, tj, ioui, iouj = data[:-num], target[:-num], \
                                               ti[:-num], tj[:-num], \
                                               ioui[:-num], iouj[:-num]
            batch_size = data.shape[0]
            #print('batcj _size ', batch_size, num)
        #print('data ', data.shape)
        num_group = batch_size // args.samples_per_class
        rnd_inds = np.random.permutation(num_group)

        # rois
        rois_0 = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                            target[:,:4].to(device)), dim=1)
        rois_i = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                            ti.to(device)), dim=1)
        rois_j = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                            tj.to(device)), dim=1)

        _, pooled_0, out0 = model(data, rois_0) # (bs, 1, 84, 84), (bs, 1600)
        _, pooled_i, _ = model(data, rois_i)
        _, pooled_j, _ = model(data, rois_j)

        loss_cls = torch.tensor(0)
        #if phase == 'Train':
        #loss_cls = criterion(out0, target[:, -1].long())

        # compute loss order
        iou = torch.stack((ioui, iouj), 1)  # (bs, 2)
        f = torch.stack((pooled_i, pooled_j)).permute(1, 0, 2)  # (bs, 2, dim)

        inds = torch.max(iou, 1).indices
        p = f[torch.arange(0, batch_size), inds, :]
        n = f[torch.arange(0, batch_size), 1 - inds, :]
        if 'shuffle' in args.train_mode:
            p = p.view(-1, args.samples_per_class, args.dim)[rnd_inds].view(-1, args.dim)
            n = n.view(-1, args.samples_per_class, args.dim)[rnd_inds].view(-1, args.dim)

        # proto should be the mean of each group
        proto = pooled_0.view(-1, args.samples_per_class, args.dim).mean(1) # (num_group, 512)
        proto = proto.unsqueeze(1).repeat(1, args.samples_per_class, 1).view(-1, args.dim)
        if  'self' in args.train_mode :
            loss_ord = triplet_loss(pooled_0, p, n) #(bs,)
        elif 'proto' in args.train_mode:
            loss_ord = triplet_loss(proto, p, n)


        loss = loss_cls + loss_ord * args.lamb
        # update model
        if phase == 'Train':
            loss.backward()
            optimizer.step()

        total_losses.append(loss.item())
        recon_losses.append(loss_cls.item())
        ord_losses.append(loss_ord.item())

        # compute accuracy self
        d_a0a1 = F.pairwise_distance(pooled_0, pooled_i)
        d_a0a2 = F.pairwise_distance(pooled_0, pooled_j)
        sub_pix = iouj - ioui
        sub_d = d_a0a1 - d_a0a2
        mask = (torch.sign(sub_d) == torch.sign(sub_pix))
        correct_inst += mask.sum().item()

        d_a0a1 = F.pairwise_distance(proto, pooled_i)
        d_a0a2 = F.pairwise_distance(proto, pooled_j)
        sub_d = d_a0a1 - d_a0a2
        mask = (torch.sign(sub_d) == torch.sign(sub_pix))
        correct_proto += mask.sum().item()
         # acc other
        pooled_i = pooled_i.view(-1, args.samples_per_class, args.dim)[rnd_inds].view(-1, args.dim)
        pooled_j = pooled_j.view(-1, args.samples_per_class, args.dim)[rnd_inds].view(-1, args.dim)
        ioui = ioui.view(-1, args.samples_per_class)[rnd_inds].view(-1)
        iouj = iouj.view(-1, args.samples_per_class)[rnd_inds].view(-1)

        d_a0a1 = F.pairwise_distance(pooled_0, pooled_i)
        d_a0a2 = F.pairwise_distance(pooled_0, pooled_j)
        sub_pix = iouj - ioui
        sub_d = d_a0a1 - d_a0a2
        mask = (torch.sign(sub_d) == torch.sign(sub_pix))
        correct_inst_other += mask.sum().item()

        d_a0a1 = F.pairwise_distance(proto, pooled_i)
        d_a0a2 = F.pairwise_distance(proto, pooled_j)
        sub_d = d_a0a1 - d_a0a2
        mask = (torch.sign(sub_d) == torch.sign(sub_pix))
        correct_proto_other += mask.sum().item()

        count += batch_size

        if batch_idx % args.log_interval == 0:
            if batch_idx == 0:
                print('\n')
            print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                phase, epoch, batch_idx * batch_size,
                len(data_loader.dataset),
                100. * batch_idx / len_dataloader, np.mean(total_losses)))

        #break

    total_losses = np.mean(total_losses)
    #iterations = args.iterations if phase == 'Train' else args.iterations_test
    #print('count ', count)
    total_img = count #iterations * args.samples_per_class * args.classes_per_set
    acc_inst = 100. * correct_inst / total_img
    acc_proto = 100. * correct_proto / total_img
    acc_inst_other = 100. * correct_inst_other / total_img
    acc_proto_other = 100. * correct_proto_other / total_img

    # plot
    # input = convert_image_np(
    #     torchvision.utils.make_grid(data[:16].cpu(), nrow=4,
    #                                 padding=5, pad_value=123), norm=False)
    # recon_img0 = TF.normalize(recon_img0, (0.1307,), (0.3081,))
    # output = convert_image_np(
    #     torchvision.utils.make_grid(recon_img0[:16].cpu().detach(), nrow=4,
    #                                 padding=5, pad_value=123),
    #         norm=False)
    # f, axarr = plt.subplots(1, 2)
    # axarr[0].imshow(input)
    # axarr[0].set_title('Dataset Images')
    # axarr[0].set_axis_off()
    #
    # axarr[1].imshow(output)
    # axarr[1].set_title('Reconstructed Images')
    # axarr[1].set_axis_off()

    log_message = {"{} Total Loss".format(phase): total_losses,
                   "{} Cls Loss".format(phase): np.mean(recon_losses),
                   "{} Ord Loss".format(phase): np.mean(ord_losses),
                   '{} Ord Acc Inst'.format(phase): acc_inst,
                   '{} Ord Acc Proto'.format(phase): acc_proto,
                   '{} Ord Acc Inst Other'.format(phase): acc_inst_other,
                   '{} Ord Acc Proto Other'.format(phase): acc_proto_other,
                   #"{} Images".format(phase): wandb.Image(plt)
                   }
    if phase == 'Train':
        current_lr = scheduler.get_last_lr()[0]
        log_message.update({'learning rate': current_lr})

    if upload:
        wandb.log(log_message, step=epoch)
    plt.close()

    return total_losses, acc_inst, acc_proto, acc_inst_other, acc_proto_other

def evaluate(upload=True):
    for name in ['last', 'best']:
        model_path = os.path.join(saveroot, '{}.pth.tar'.format(name))
        # load model
        print('loading net ckpt from ', model_path)
        ckpt = torch.load(model_path)
        epoch = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        # acc_proto = ckpt['acc']
        # print(("=> loaded model checkpoint epoch {} {}\n\tacc {:.2f}".format(epoch, name, acc_proto)))
        acc_inst, acc_proto, acc_inst_other, \
        acc_proto_other = ckpt['acc_inst'], ckpt['acc_proto'],\
                          ckpt['acc_inst_other'], ckpt['acc_proto_other']
        print(("=> loaded model checkpoint epoch {} {}\n\tacc_inst {:.2f}"
               "\tacc_proto {:.2f}\tacc_inst_other {:.2f}"
               "\tacc_proto_other {:.2f}".format(epoch, name, acc_inst, acc_proto,
                                                acc_inst_other, acc_proto_other)))
        if upload:
            wandb.run.summary.update({
            'Test {} Ord Acc Inst'.format(name.capitalize()): acc_inst,
            'Test {} Ord Acc Proto'.format(name.capitalize()): acc_proto,
            'Test {} Ord Acc Inst Other'.format(name.capitalize()): acc_inst_other,
            'Test {} Ord Acc Proto Other'.format(name.capitalize()): acc_proto_other,
            })

if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['WANDB_NAME'] = args.savename
    wandb.init(project="selfpaced")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saveroot = os.path.join('/research/cbim/vast/tl601/results/mnist/',
                       args.savename)
    os.makedirs(saveroot, exist_ok=True)

    train_loader, test_loader = init_dataloader()

    model = Net(pooling_size=args.pooling_size, pretrained=True, dim=args.dim,
                        pooling_mode=args.pooling_mode).to(device)
    wandb.watch(model, log_freq=10)
    wandb.config.update(args)

    criterion = nn.CrossEntropyLoss()
    triplet_loss = nn.TripletMarginLoss(margin=args.margin, p=2)
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=args.steps, gamma=0.1)

    if args.evaluate == 1:
        epoch =0
        evaluate()
        # data = next(iter(test_loader))
        # data = data[0][:16]
        # output = model(data.to(device))
        # # data = TF.normalize(data, (0.1307,), (0.3081,))
        # output = TF.normalize(output, (0.1307,), (0.3081,))
        # input = convert_image_np(
        #     torchvision.utils.make_grid(data, nrow=4), norm=False)
        # output = convert_image_np(
        #     torchvision.utils.make_grid(output.cpu().detach(), nrow=4), norm=False)
        # f, axarr = plt.subplots(1, 2)
        # axarr[0].imshow(input)
        # axarr[0].set_title('Dataset Images')
        # axarr[0].set_axis_off()
        # axarr[1].imshow(output)
        # axarr[1].set_title('Reconstructed Images')
        # axarr[1].set_axis_off()
        # plt.show()
        exit()

    best_acc = 0.
    save_model = False
    no_improve_epoch = 0
    for epoch in range(args.epochs):
        train_log = feed_data(model, data_loader=train_loader, is_train=True)
        scheduler.step()

        with torch.no_grad():
            test_loss, test_acc_inst, test_acc_proto, test_acc_inst_other,\
            test_acc_proto_other = feed_data(model,
                                            data_loader=test_loader, is_train=False)

            # save model
            if test_acc_proto > best_acc:
                save_model = True
                best_acc = test_acc_proto
                no_improve_epoch = 0
            else:
                save_model = False
                no_improve_epoch += 1
            if save_model:
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'acc_inst': test_acc_inst,
                    'acc_proto': test_acc_proto,
                    'acc_inst_other': test_acc_inst_other,
                    'acc_proto_other': test_acc_proto_other
                }, os.path.join(saveroot,'best.pth.tar'))

            # break training
            if no_improve_epoch > args.patience:
                print('stop training...')
                break

        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'acc_inst': test_acc_inst,
            'acc_proto': test_acc_proto,
            'acc_inst_other': test_acc_inst_other,
            'acc_proto_other': test_acc_proto_other
        }, os.path.join(saveroot, 'last.pth.tar'))

    evaluate()



"""
Train autoencoder+ordinal on clutter omniglot, using few shot setting
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
from util.augmentations import Compose
from util.data_aug import Resize
from torchvision import transforms
from util.utils import generate_boxes
from datasets.clutter_omg_fewshot import OmniglotDataset, init_sampler
from models.mnist_ae_ord import AutoencoderProj
import matplotlib.pyplot as plt
from util.utils import convert_image_np

DEBUG = False

def init_dataloader():
    anchors = generate_boxes(base_size=4,
                             feat_height=21, feat_width=21, img_size=84,
                             feat_stride=4,
                             ratios=[1.0],
                             min_box_side=28,
                             scales=np.array(range(7, 20)))

    trainset = OmniglotDataset(anchors=anchors, mode='train',
                               bg_name=args.bg_name,
                               root='/research/cbim/vast/tl601/Dataset/Omniglot')
    testset = OmniglotDataset(anchors=anchors, mode='val',
                              bg_name=args.bg_name,
                              root='/research/cbim/vast/tl601/Dataset/Omniglot')

    train_batchsampler = init_sampler(args.classes_per_set, args.samples_per_class * 2,
                                    trainset.y[:, -1], args.iterations, 'train')
    test_batchsampler = init_sampler(args.classes_per_set, args.samples_per_class * 2,
                                      testset.y[:, -1], args.iterations_test, 'val')
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_sampler=train_batchsampler)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_sampler=test_batchsampler)

    return train_loader, test_loader

def feed_data(model, data_loader, is_train, upload=True):
    if is_train:
        phase = 'Train'
        model.train()
    else:
        phase = 'Test'
        model.eval()

    total_losses, recon_losses, ord_losses = [], [], []
    correct_inst, correct_proto = 0, 0
    len_dataloader = len(data_loader)
    for batch_idx, (data, target, boxes, ious) in enumerate(data_loader):
        batch_size = data.shape[0] # bs = N * 2K
        data, target, boxes, ious = data.to(device), target.to(device).float(), \
                                    boxes.to(device), ious.to(device)
        target_org = target.clone()
        if phase == 'Train':
            optimizer.zero_grad()

        '''split to support set and target set '''
        target_data = data.view(args.classes_per_set, args.samples_per_class * 2,
                                1, args.img_size, args.img_size)[:, args.samples_per_class:, :, :, :]  # (N,K,1,84,84)
        boxes = boxes.view(args.classes_per_set, args.samples_per_class * 2, 2, 4)  # (N, 2K, 2, 4)
        ious = ious.view(args.classes_per_set, args.samples_per_class * 2, 2)  # (N, 2K, 2)
        target_boxes = boxes[:, args.samples_per_class:, :, :]  # (N,K,2,4)
        target_ious = ious[:, args.samples_per_class:, :].reshape(-1, 2)  # (N,K,2)

        # rois
        rois_targets = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                                  target[:, :4]), dim=1)  # (2NK, 5)
        rois = torch.cat((torch.arange(0, batch_size // 2).unsqueeze(1).repeat(1, 2).view(-1, 1).to(device),
                          target_boxes.reshape(-1, 4)), dim=1)  # (NK2, 5)

        rec_img, pooled_gt_all = model(data, rois_targets)  # (bs, 1, 84, 84), (bs, dim)
        _, pooleds = model(target_data.reshape(-1, 1, args.img_size, args.img_size), rois)  # (N*K*2, dim)

        pooled_gt_all = pooled_gt_all.view(args.classes_per_set, args.samples_per_class * 2, -1)
        pooled_gt_supp, pooled_gt_tar = pooled_gt_all[:, :args.samples_per_class, :], \
                                        pooled_gt_all[:, args.samples_per_class:, :]  # (N,K, dim)

        '''compute loss order using support proto'''
        f = pooleds.view(args.classes_per_set * args.samples_per_class, 2, -1)  # (N*K, 2, dim)
        inds = torch.max(target_ious.view(-1, 2), 1).indices
        p, n = f[torch.arange(0, batch_size // 2), inds, :], f[torch.arange(0, batch_size // 2), 1 - inds, :]
        proto = pooled_gt_supp.mean(1)  # (N, 512)
        proto = proto.unsqueeze(1).repeat(1, args.samples_per_class, 1).view(-1, 512)  # (N*K, dim)

        if args.train_mode == 'self':
            loss_ord = triplet_loss(pooled_gt_tar.reshape(args.classes_per_set * args.samples_per_class, 512),
                                    p, n)  # (bs,)
        elif args.train_mode == 'proto':
            loss_ord = triplet_loss(proto, p, n)  # (bs,)

        '''compute reconstruction loss'''
        loss_recon = criterion(rec_img, data)

        loss = loss_recon + loss_ord * args.lamb
        # update model
        if phase == 'Train':
            loss.backward()
            optimizer.step()

        total_losses.append(loss.item())
        recon_losses.append(loss_recon.item())
        ord_losses.append(loss_ord.item())

        # compute accuracy self
        pooled_gt_tar = pooled_gt_tar.reshape(args.classes_per_set * args.samples_per_class, -1)
        d_a0a1 = F.pairwise_distance(pooled_gt_tar, f[torch.arange(0, batch_size // 2), 0, :])
        d_a0a2 = F.pairwise_distance(pooled_gt_tar, f[torch.arange(0, batch_size // 2), 1, :])
        sub_pix = target_ious[:, 1] - target_ious[:, 0]
        sub_d = d_a0a1 - d_a0a2
        mask = (torch.sign(sub_d) == torch.sign(sub_pix))
        correct_inst += mask.sum().item()

        d_a0a1 = F.pairwise_distance(proto, f[torch.arange(0, batch_size // 2), 0, :])
        d_a0a2 = F.pairwise_distance(proto, f[torch.arange(0, batch_size // 2), 1, :])
        sub_d = d_a0a1 - d_a0a2
        mask = (torch.sign(sub_d) == torch.sign(sub_pix))
        correct_proto += mask.sum().item()

        if batch_idx % args.log_interval == 0:
            if batch_idx == 0:
                print('\n')
            print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                phase, epoch, batch_idx * batch_size,
                len(data_loader.dataset),
                              100. * batch_idx / len_dataloader, np.mean(total_losses)))

        #break


    total_losses = np.mean(total_losses)
    iterations = args.iterations if phase == "Train" else args.iterations_test
    total_img = args.iterations * args.samples_per_class * args.classes_per_set
    acc_inst = 100. * correct_inst / total_img
    acc_proto = 100. * correct_proto / total_img

    # plot
    input = convert_image_np(
        torchvision.utils.make_grid(data[:16].cpu(), nrow=4,
                                    padding=5, pad_value=123), norm=False)
    recon_img0 = TF.normalize(rec_img, (0.1307,), (0.3081,))
    output = convert_image_np(
        torchvision.utils.make_grid(recon_img0[:16].cpu().detach(), nrow=4,
                                    padding=5, pad_value=123),
            norm=False)
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(input)
    axarr[0].set_title('Dataset Images')
    axarr[0].set_axis_off()

    axarr[1].imshow(output)
    axarr[1].set_title('Reconstructed Images')
    axarr[1].set_axis_off()

    log_message = {"{} Total Loss".format(phase): total_losses,
                   "{} Recon Loss".format(phase): np.mean(recon_losses),
                   "{} Ord Loss".format(phase): np.mean(ord_losses),
                   '{} Ord Acc Inst'.format(phase): acc_inst,
                   '{} Ord Acc Proto'.format(phase): acc_proto,
                   "{} Images".format(phase): wandb.Image(plt)
                   }
    if phase == 'Train':
        current_lr = scheduler.get_last_lr()[0]
        log_message.update({'learning rate': current_lr})

    if upload:
        wandb.log(log_message, step=epoch)
    plt.close()

    return total_losses, acc_inst, acc_proto

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
        acc_inst, acc_proto = ckpt['acc_inst'], ckpt['acc_proto']
        print(("=> loaded model checkpoint epoch {} {}\n\tacc_inst {:.2f}"
               "\tacc_proto {:.2f}".format(epoch, name, acc_inst, acc_proto)))
        if upload:
            wandb.run.summary.update({
            'Test {} Ord Acc Inst'.format(name.capitalize()): acc_inst,
            'Test {} Ord Acc Proto'.format(name.capitalize()): acc_proto,
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

    model = AutoencoderProj(pooling_size=args.pooling_size,
                        pooling_mode=args.pooling_mode).to(device)
    wandb.watch(model, log_freq=10)
    wandb.config.update(args)

    criterion = nn.MSELoss()  # mean square error loss
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
            test_loss, test_acc_inst, test_acc_proto = feed_data(model,
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
    }, os.path.join(saveroot, 'last.pth.tar'))

    evaluate()



"""
Train autoencoder+ordinal on digit 4 of clutter mnist, using few shot setting
https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
Created on 4/27/2021 2:54 PM

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
from datasets.clutter_mnist_scale_anchor import MNIST_CoLoc
from models.mnist_ae_ord import Autoencoder, AutoencoderProj
import matplotlib.pyplot as plt
from util.utils import convert_image_np

DEBUG = False

def init_dataloader():
    kwargs = {'num_workers': 8, 'pin_memory': True}

    train_transform = Compose([Resize(84)])
    test_transform = Compose([Resize(84)])
    # for 28*28
    anchors = generate_boxes(base_size=4,
                             feat_height=21, feat_width=21, img_size=84,
                             feat_stride=4,
                             ratios=[1.0],
                             min_box_side=28,
                             scales=np.array(range(7, 20)))
    # for tight box
    # anchors = generate_boxes(base_size=4,
    #                          feat_height=21, feat_width=21, img_size=84,
    #                          feat_stride=4,
    #                          ratios=np.linspace(0.5, 3.0, num=10),
    #                          min_box_side=20,
    #                          scales=np.array(range(7, 20)))
    print('number of anchors for 84*84 image ', anchors.shape[0])
    trainset = MNIST_CoLoc(root='.', train=True, digit=args.digit,
                           anchors=anchors, bg_name=args.bg_name,
                           sample_size=args.sample_size,
                           datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                           clutter=1, transform=train_transform,
                           )
    testset = MNIST_CoLoc(root='.', train=False, digit=args.digit,
                          anchors=anchors, bg_name=args.bg_name,
                          datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                          clutter=1, transform=test_transform
                          )

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    print('total train image: ', len(train_loader.dataset), ' test image: ',
          len(test_loader.dataset))
    return train_loader, test_loader

def feed_data(model, data_loader, is_train, upload=True):
    if is_train:
        phase = 'Train'
        model.train()
    else:
        phase = 'Test'
        model.eval()

    count = 0
    total_losses, recon_losses, ord_losses = [], [], []
    correct_inst, correct_proto, correct_inst_other, correct_proto_other = 0, 0, 0, 0
    len_dataloader = len(data_loader)
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
            # print('batch_size ', batch_size)
            # print('num ', num)
            data, target, ti, tj, ioui, iouj = data[:-num], target[:-num], \
                                               ti[:-num], tj[:-num], \
                                               ioui[:-num], iouj[:-num]
            batch_size = data.shape[0]

        num_group = batch_size // args.samples_per_class
        rnd_inds = np.random.permutation(num_group)
        if num_group == 0:
            continue
        count += batch_size

        # rois
        rois_0 = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                            target.to(device)), dim=1)
        rois_i = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                            ti.to(device)), dim=1)
        rois_j = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                            tj.to(device)), dim=1)

        recon_img0, pooled_0 = model(data, rois_0) # (bs, 1, 84, 84), (bs, 1600)
        recon_imgi, pooled_i = model(data, rois_i)
        recon_imgj, pooled_j = model(data, rois_j)

        loss_recon = criterion(recon_img0, data)
        #loss = F.smooth_l1_loss(recon, img)

        # compute loss order
        iou = torch.stack((ioui, iouj), 1)  # (bs, 2)
        f = torch.stack((pooled_i, pooled_j)).permute(1, 0, 2)  # (bs, 2, dim)

        inds = torch.max(iou, 1).indices
        p = f[torch.arange(0, batch_size), inds, :].view(-1,
                                        args.samples_per_class, 512)[rnd_inds].view(-1, 512)
        n = f[torch.arange(0, batch_size), 1 - inds, :].view(-1,
                                        args.samples_per_class, 512)[rnd_inds].view(-1, 512)
        #proto = torch.mean(pooled_0, dim=0).view(1, -1).repeat(batch_size, 1)
        # proto should be the mean of each group
        proto = pooled_0.view(-1, args.samples_per_class, 512).mean(1) # (num_group, 512)
        proto = proto.unsqueeze(1).repeat(1, args.samples_per_class, 1).view(-1, 512)
        if args.train_mode == 'shuffle_proto':
            loss_ord = triplet_loss(proto, p, n)
        elif args.train_mode == 'shuffle_self':
            loss_ord = triplet_loss(pooled_0, p, n) #(bs,)

        loss = loss_recon + loss_ord * args.lamb
        # update model
        if phase == 'Train':
            loss.backward()
            optimizer.step()

        total_losses.append(loss.item())
        recon_losses.append(loss_recon.item())
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
        pooled_i = pooled_i.view(-1, args.samples_per_class, 512)[rnd_inds].view(-1, 512)
        pooled_j = pooled_j.view(-1, args.samples_per_class, 512)[rnd_inds].view(-1, 512)
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

        if batch_idx % args.log_interval == 0:
            if batch_idx == 0:
                print('\n')
            print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                phase, epoch, batch_idx * batch_size,
                len(data_loader.dataset),
                100. * batch_idx / len_dataloader, np.mean(total_losses)))

        #break
    #print('count ', count, ' len(data) ', len(data_loader.dataset))
    total_losses = np.mean(total_losses)
    acc_inst = 100. * correct_inst / count #len(data_loader.dataset)
    acc_proto = 100. * correct_proto / count #len(data_loader.dataset)
    acc_inst_other = 100. * correct_inst_other / count #len(data_loader.dataset)
    acc_proto_other = 100. * correct_proto_other / count #len(data_loader.dataset)

    # plot
    data, target, ti, tj, ioui, iouj = next(iter(data_loader))
    input = convert_image_np(
        torchvision.utils.make_grid(data[:16].cpu(), nrow=4,
                                    padding=5, pad_value=123), norm=False)
    recon_img0 = TF.normalize(recon_img0, (0.1307, ), (0.3081, ))
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
                   '{} Ord Acc Inst Other'.format(phase): acc_inst_other,
                   '{} Ord Acc Proto Other'.format(phase): acc_proto_other,
                   "{} Images".format(phase): wandb.Image(plt)
                   }
    if phase == 'Train':
        current_lr = scheduler.get_last_lr()[0]
        log_message.update({'learning rate': current_lr})

    if upload:
        wandb.log(log_message, step=epoch)
    plt.close()

    return total_losses, acc_inst, acc_proto, acc_inst_other, acc_proto_other

def evaluate(upload=True):
    anchors = generate_boxes(base_size=4,
                             feat_height=21, feat_width=21, img_size=84,
                             feat_stride=4,
                             ratios=[1.0],
                             min_box_side=28,
                             scales=np.array(range(7, 20)))
    kwargs = {'num_workers': 8, 'pin_memory': True}
    test_transform = Compose([Resize(84)])
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
        test_accs_inst, test_accs_proto, test_accs_inst_other, \
        test_accs_proto_other = [], [], [], []
        with torch.no_grad():
            for d in range(0, 10):
                # if d == 4:
                #     continue
                testset = MNIST_CoLoc(root='.', train=False, digit=d,
                                      anchors=anchors, bg_name=args.bg_name,
                                      datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                                      clutter=1, transform=test_transform
                                      )
                test_loader = torch.utils.data.DataLoader(
                    testset, batch_size=args.batch_size, shuffle=False, **kwargs)
                test_loss, test_acc_inst, test_acc_proto, test_acc_inst_other, \
                test_acc_proto_other = feed_data(model,
                                            data_loader=test_loader, is_train=False, upload=False)
                if d != args.digit:
                    test_accs_inst.append(test_acc_inst)
                    test_accs_proto.append(test_acc_proto)
                    test_accs_inst_other.append(test_acc_inst_other)
                    test_accs_proto_other.append(test_acc_proto_other)
                print(d, ' test_acc_inst {:.2f}\ttest_acc_proto {:.2f}'
                         '\ttest_acc_inst_other {:.2f}\ttest_acc_proto_other {:.2f}'.format(
                      test_acc_inst, test_acc_proto, test_acc_inst_other,
                      test_acc_proto_other))
            print('mean test_acc_inst {:.2f}\ttest_acc_proto {:.2f}'
                     '\ttest_acc_inst_other {:.2f}\ttest_acc_proto_other {:.2f}'.format(
                np.mean(test_accs_inst), np.mean(test_accs_proto), np.mean(test_accs_inst_other),
                np.mean(test_accs_proto_other)))
        if upload:
            wandb.run.summary.update({
            'Test {} Ord Acc Inst'.format(name.capitalize()): acc_inst,
            'Test {} Ord Acc Proto'.format(name.capitalize()): acc_proto,
            'Test {} Ord Acc Inst Other'.format(name.capitalize()): acc_inst_other,
            'Test {} Ord Acc Proto Other'.format(name.capitalize()): acc_proto_other,
            'Test {} Gen Ord Acc Inst'.format(name.capitalize()): np.mean(test_accs_inst),
            'Test {} Gen Ord Acc Proto'.format(name.capitalize()): np.mean(test_accs_proto),
            'Test {} Gen Ord Acc Inst Other'.format(name.capitalize()): np.mean(test_accs_inst_other),
            'Test {} Gen Ord Acc Proto Other'.format(name.capitalize()): np.mean(test_accs_proto_other),
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

    model = AutoencoderProj(channel=1, pooling_size=args.pooling_size,
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
        # model_path = os.path.join(saveroot, 'last.pth.tar')
        # # load model
        # print('loading net ckpt from ', model_path)
        # ckpt = torch.load(model_path)
        # epoch = ckpt['epoch']
        # model.load_state_dict(ckpt['state_dict'])
        # acc_inst = ckpt['acc']
        # print(("=> loaded model checkpoint epoch {} {}\n\tacc {:.2f}".format(epoch, 'last', acc_inst)))
        # total_losses, acc_inst, acc_proto, acc_inst_other, acc_proto_other = \
        #     feed_data(model, data_loader=train_loader, is_train=False, upload=True)
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
        }, os.path.join(saveroot, '{}.pth.tar'.format(epoch)))

    evaluate()



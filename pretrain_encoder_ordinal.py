# -*- coding: utf-8 -*-
# mnist bg_name ['clean','clutter', 'patch', 'gaussian_noise', 'impulse_noise']
# cub bg_name ['warbler', 'gull', 'gull_59_64.json', 'wren', 'sparrow', 'oriole', 'kingfisher', 'vireo']
# @Time : 9/3/21 4:57 PM
# @Author : Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
import os, wandb, torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
from opts import parser
from util.augmentations import Compose
from util import joint_transforms as t
from util.data_aug import Resize
from util.utils import generate_boxes, convert_image_np, calculate_correct
from datasets.clutter_mnist import MNIST_Corrupted
from datasets.cub import CUB
from datasets.coco_onecls import CocoDataset
from models.mnist_model import AutoencoderProj
from models.cub_model import Encoder_CUB

def init_dataloader():
    kwargs = {'num_workers': 8, 'pin_memory': True}
    if args.dataset == 'mnist':
        print('number of anchors for {}*{} image {} '.format(args.img_size, args.img_size, anchors.shape[0]))
        trainset = MNIST_Corrupted(root='.', train=True, digit=args.digit,
                               anchors=anchors, bg_name=args.bg_name,
                               sample_size=args.sample_size,
                               datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist_github',
                               transform=transform
                               )
        testset = MNIST_Corrupted(root='.', train=False, digit=args.digit,
                              anchors=anchors, bg_name=args.bg_name,
                              datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist_github',
                              transform=transform
                              )
    elif args.dataset == 'cub':
        assert (args.img_size == 224)
        trainset = CUB('/research/cbim/vast/tl601/Dataset/CUB_200_2011',
                       anchors=anchors, mode='base', bg_name=args.bg_name,
                       img_size=args.img_size, transform=transform)
        testset = CUB('/research/cbim/vast/tl601/Dataset/CUB_200_2011',
                      anchors=anchors, mode='val', bg_name=args.bg_name,
                      img_size=args.img_size, transform=transform)
    elif args.dataset == 'coco':
        assert (args.img_size == 224)
        trainset = CocoDataset(root='/research/cbim/vast/tl601/Dataset/coco/train2017',
                               annFile='/research/cbim/vast/tl601/Dataset/coco/annotations/instances_train2017.json',
                               anchors=anchors, img_size=args.img_size, selected_cls=args.sel_cls, transforms=transform)
        testset = CocoDataset(root='/research/cbim/vast/tl601/Dataset/coco/val2017',
                              annFile='/research/cbim/vast/tl601/Dataset/coco/annotations/instances_val2017.json',
                              anchors=anchors, img_size=args.img_size, selected_cls=args.sel_cls, transforms=transform)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    print('total train image: ', len(train_loader.dataset), ' test image: ',
          len(test_loader.dataset))
    return train_loader, test_loader

def init_model():
    if args.dataset == 'mnist':
        model = AutoencoderProj(channel=1, pooling_size=args.pooling_size, dim=args.dim,
                                pooling_mode=args.pooling_mode).to(device)
    elif args.dataset == 'cub' or args.dataset == 'coco':
        model = Encoder_CUB(pooling_size=args.pooling_size, pretrained=True, base=args.backbone,
                            dim=args.dim, pooling_mode=args.pooling_mode).to(device)

    return model

def feed_data(model, data_loader, is_train, upload=True):
    if is_train:
        phase = 'Train'
        model.train()
    else:
        phase = 'Test'
        model.eval()

    count = 0
    total_losses, base_losses, ord_losses = [], [], []
    correct_inst, correct_proto, correct_inst_other, correct_proto_other = 0, 0, 0, 0
    len_dataloader = len(data_loader)
    for batch_idx, (data, target, ti, tj, ioui, iouj) in enumerate(data_loader):
        batch_size = data.shape[0]
        data, target, ti, tj, ioui, iouj = data.to(device), target.to(device).float(), \
                                           ti.to(device), tj.to(device), \
                                           ioui.to(device), iouj.to(device)
        if phase == 'Train':
            optimizer.zero_grad()

        '''split to support set and target set '''
        num = batch_size % args.samples_per_class
        if num != 0:
            data, target, ti, tj, ioui, iouj = data[:-num], target[:-num], \
                                               ti[:-num], tj[:-num], \
                                               ioui[:-num], iouj[:-num]
            batch_size = data.shape[0]

        num_group = batch_size // args.samples_per_class
        if num_group == 0:
            continue
        rnd_inds = np.random.permutation(num_group)
        count += batch_size
        ''''''

        # rois
        rois   = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                            target[:, :4].to(device)), dim=1)
        rois_i = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                            ti.to(device)), dim=1)
        rois_j = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                            tj.to(device)), dim=1)

        outputs   = model(data, rois) # (recon_img, top_feat) or (pooled_feat, top_feat, class)
        outputs_i = model(data, rois_i)
        outputs_j = model(data, rois_j)

        loss_base = torch.tensor(0)
        if args.dataset == 'mnist':
            loss_base = criterion(outputs[0], data) # reconstruction loss

        '''ordinal loss'''
        iou  = torch.stack((ioui, iouj), 1) # (bs, 2)
        feat = torch.stack((outputs_i[1], outputs_j[1])).permute(1, 0, 2) # (bs, 2, dim)
        inds = torch.max(iou, 1).indices
        pos, neg = feat[torch.arange(0, batch_size), inds, :], \
                   feat[torch.arange(0, batch_size), 1-inds, :]
        if 'shuffle' in args.train_mode:
            pos = pos.view(-1, args.samples_per_class, args.dim)[rnd_inds].view(-1, args.dim)
            neg = neg.view(-1, args.samples_per_class, args.dim)[rnd_inds].view(-1, args.dim)
        # proto should be the mean of each group
        proto = outputs[1].view(-1, args.samples_per_class, args.dim).mean(1) # (num_group, args.dim)
        proto = proto.unsqueeze(1).repeat(1, args.samples_per_class, 1).view(-1, args.dim)
        if  'self' in args.train_mode :
            loss_ord = triplet_loss(outputs[1], pos, neg) #(bs,)
        elif 'proto' in args.train_mode:
            loss_ord = triplet_loss(proto, pos, neg)
        ''''''

        loss = loss_base * args.lamb_base + loss_ord * args.lamb

        if phase == 'Train':
            loss.backward()
            optimizer.step()

        total_losses.append(loss.item())
        base_losses.append(loss_base.item())
        ord_losses.append(loss_ord.item())

        '''compute accuracies'''
        diff_iou = iouj - ioui
        diff_feat = F.pairwise_distance(outputs[1], outputs_i[1]) - \
                    F.pairwise_distance(outputs[1], outputs_j[1])
        correct_inst += calculate_correct(diff_feat, diff_iou)

        diff_feat = F.pairwise_distance(proto, outputs_i[1]) - \
                    F.pairwise_distance(proto, outputs_j[1])
        correct_proto += calculate_correct(diff_feat, diff_iou)
        # acc other
        top_i = outputs_i[1].view(-1, args.samples_per_class, args.dim)[rnd_inds].view(-1, args.dim)
        top_j = outputs_j[1].view(-1, args.samples_per_class, args.dim)[rnd_inds].view(-1, args.dim)
        ioui = ioui.view(-1, args.samples_per_class)[rnd_inds].view(-1)
        iouj = iouj.view(-1, args.samples_per_class)[rnd_inds].view(-1)

        diff_feat = F.pairwise_distance(outputs[1], top_i) - \
                    F.pairwise_distance(outputs[1], top_j)
        diff_iou = iouj - ioui
        correct_inst_other += calculate_correct(diff_feat, diff_iou)

        diff_feat = F.pairwise_distance(proto, top_i) - \
                    F.pairwise_distance(proto, top_j)
        correct_proto_other += calculate_correct(diff_feat, diff_iou)
        ''''''

        if batch_idx % args.log_interval == 0:
            if batch_idx == 0:
                print('\n')
            print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                phase, epoch, batch_idx * batch_size,
                len(data_loader.dataset),
                              100. * batch_idx / len_dataloader, np.mean(total_losses)))


    total_losses = np.mean(total_losses)
    acc_inst, acc_proto, acc_inst_other, acc_proto_other = 100. * correct_inst / count, \
                                                           100. * correct_proto / count, \
                                                           100. * correct_inst_other / count, \
                                                           100. * correct_proto_other / count

    log_message = {"{} Total Loss".format(phase): total_losses,
                   "{} Base Loss".format(phase): np.mean(base_losses),
                   "{} Ord Loss".format(phase): np.mean(ord_losses),
                   '{} Ord Acc Inst'.format(phase): acc_inst,
                   '{} Ord Acc Proto'.format(phase): acc_proto,
                   '{} Ord Acc Inst Other'.format(phase): acc_inst_other,
                   '{} Ord Acc Proto Other'.format(phase): acc_proto_other}
    if args.dataset == 'mnist':
        '''plot'''
        data, target, ti, tj, ioui, iouj = next(iter(data_loader))
        input = convert_image_np(
            torchvision.utils.make_grid(data[:16].cpu(), nrow=4,
                                        padding=5, pad_value=123), norm=False)
        recon_img0 = TF.normalize(outputs[0], (0.1307,), (0.3081,))
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

        log_message.update({"{} Images".format(phase): wandb.Image(plt)})

    if phase == 'Train':
        current_lr = scheduler.get_last_lr()[0]
        log_message.update({'learning rate': current_lr})
    if upload:
        wandb.log(log_message, step=epoch)
    if args.dataset == 'mnist':
        plt.close()

    return total_losses, acc_inst, acc_proto, acc_inst_other, acc_proto_other

def evaluate(upload=True):
    kwargs = {'num_workers': 8, 'pin_memory': True}
    for name in ['last', 'best']:
        model_path = os.path.join(saveroot, '{}.pth.tar'.format(name))
        # load model
        print('loading net ckpt from ', model_path)
        ckpt = torch.load(model_path)
        epoch = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        acc_inst, acc_proto, acc_inst_other, \
        acc_proto_other = ckpt['acc_inst'], ckpt['acc_proto'], \
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
        if args.dataset == 'mnist':
            test_accs_inst, test_accs_proto, test_accs_inst_other, \
            test_accs_proto_other = [], [], [], []
            with torch.no_grad():
                for d in range(0, 10):
                    # if d == args.digit:
                    #     continue
                    testset = MNIST_Corrupted(root='.', train=False, digit=d,
                                          anchors=anchors, bg_name=args.bg_name,
                                          datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist_github',
                                          transform=transform
                                          )
                    test_loader = torch.utils.data.DataLoader(
                        testset, batch_size=args.batch_size, shuffle=False, **kwargs)
                    test_loss, test_acc_inst, test_acc_proto, test_acc_inst_other, \
                    test_acc_proto_other = feed_data(model, data_loader=test_loader, is_train=False, upload=False)
                    if d != args.digit:
                        test_accs_inst.append(test_acc_inst)
                        test_accs_proto.append(test_acc_proto)
                        test_accs_inst_other.append(test_acc_inst_other)
                        test_accs_proto_other.append(test_acc_proto_other)

                    print(d, ' test_acc_inst {:.2f}\ttest_acc_proto {:.2f}'
                             '\ttest_acc_inst_other {:.2f}\ttest_acc_proto_other {:.2f}'.format(
                        test_acc_inst, test_acc_proto, test_acc_inst_other, test_acc_proto_other))
            print('mean test_acc_inst {:.2f}\ttest_acc_proto {:.2f}'
                  '\ttest_acc_inst_other {:.2f}\ttest_acc_proto_other {:.2f}'.format(
                np.mean(test_accs_inst), np.mean(test_accs_proto), np.mean(test_accs_inst_other),
                np.mean(test_accs_proto_other)))
            if upload:
                wandb.run.summary.update({
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
    saveroot = os.path.join('/research/cbim/vast/tl601/results/loc-agent/',
                            args.savename)
    os.makedirs(saveroot, exist_ok=True)

    '''set anchors'''
    if args.dataset == 'mnist':
        anchors = generate_boxes(base_size=4,
                                 feat_height=21, feat_width=21, img_size=args.img_size,
                                 feat_stride=4,
                                 ratios=[1.0],
                                 min_box_side=28,
                                 scales=np.array(range(7, 20)))
        transform = Compose([Resize(args.img_size)])
    elif args.dataset == 'cub' or args.dataset == 'coco':
        anchors = generate_boxes(base_size=16, feat_height=14, feat_width=14,
                                 min_box_side=25, img_size=args.img_size,
                                 feat_stride=16, ratios=np.linspace(0.3, 3.5, num=15),
                                 scales=np.array(range(2, 13)))
        transform = t.Compose([
            t.ConvertFromPIL(),
            t.ToPercentCoords(),
            t.Resize(args.img_size),
            t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            t.ToTensor()  # no change to (0, 1)
        ])
    train_loader, test_loader = init_dataloader()

    model = init_model()
    wandb.watch(model, log_freq=10)
    wandb.config.update(args)

    if args.dataset == 'mnist':
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

    if args.evaluate:
        epoch = 0
        evaluate()
        exit()

    best_acc = 0.
    save_model = False
    no_improve_epoch = 0
    for epoch in range(args.epochs):
        train_log = feed_data(model, data_loader=train_loader, is_train=True)
        scheduler.step()

        with torch.no_grad():
            test_loss, test_acc_inst, test_acc_proto, test_acc_inst_other, \
            test_acc_proto_other = feed_data(model, data_loader=test_loader, is_train=False)

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


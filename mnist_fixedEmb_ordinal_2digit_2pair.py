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
from datasets.clutter_mnist_scale_anchor_2digit_2p import MNIST_CoLoc
from models.mnist_ae_ord import Autoencoder, AutoencoderProj
from model.roi_layers import ROIPool
from models.mnist_embed import EmbeddingNet
import matplotlib.pyplot as plt
from util.utils import convert_image_np
from losses.contrastive import OnlineContrastiveLoss
from losses.selectors import AllPositivePairSelector, HardPositivePairSelector, HardNegativePairSelector

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

def feed_data(model, embed_net, data_loader, is_train, upload=True):
    if is_train:
        phase = 'Train'
        model.train()
    else:
        phase = 'Test'
        model.eval()

    count = 0
    total_losses, recon_losses, ord_losses = [], [], []
    ord_p1_losses, ord_p2_losses = [], []
    correct_inst_fg, correct_proto_fg = 0, 0
    correct_inst_bg, correct_proto_bg = 0, 0
    len_dataloader = len(data_loader)
    for batch_idx, (data, target, boxes, ious) in enumerate(data_loader):
        batch_size = data.shape[0]
        data, target, boxes, ious = data.to(device), target.to(device).float(), \
                                           boxes.to(device), ious.to(device)
        '''split to support set and target set '''
        num = batch_size % args.samples_per_class
        if num != 0:
            data, target, boxes, ious = data[:-num], target[:-num], \
                                               boxes[:-num], ious[:-num]

            batch_size = data.shape[0]

        pos_ious, neg_ious = ious[:, :2], ious[:, 2:]

        if phase == 'Train':
            optimizer.zero_grad()

        count += batch_size
        num_group = batch_size // args.samples_per_class
        rnd_inds = np.random.permutation(num_group)

        # rois
        rois_targets = torch.cat((torch.arange(0, batch_size).unsqueeze(1).repeat(1,2).view(-1, 1).to(device),
                            target.view(-1, 5)[:, :4]), dim=1) # (2bs, 5)
        rois = torch.cat((torch.arange(0, batch_size).float().unsqueeze(1).repeat(1,4).view(-1, 1).to(device),
                            boxes.view(-1, 4)), dim=1) # (4bs, 5)

        recon_img, pooled_target = model(data, rois_targets) # (bs, 1, 84, 84), (2bs, 1600)
        pos_pooled, neg_pooled = pooled_target.view(batch_size, 2, -1)[:, 0], \
                                 pooled_target.view(batch_size, 2, -1)[:, 1]
        _, pooleds = model(data, rois)  # (bs, 1, 84, 84), (4bs, 1600)
        pooleds = pooleds.view(batch_size, 4, -1)

        loss_recon = criterion(recon_img, data)

        '''compute loss order'''
        pos_f = pooleds[:, :2] # (bs, 2, dim)
        inds = torch.max(pos_ious, 1).indices
        pos_p = pos_f[torch.arange(0, batch_size), inds, :].view(-1,
                         args.samples_per_class, args.dim)[rnd_inds].view(-1, args.dim)
        pos_n = pos_f[torch.arange(0, batch_size), 1-inds, :].view(-1,
                         args.samples_per_class, args.dim)[rnd_inds].view(-1, args.dim)

        neg_f = pooleds[:, 2:] # (bs, 2, dim)
        inds = torch.max(neg_ious, 1).indices
        neg_p = neg_f[torch.arange(0, batch_size), inds, :].view(-1,
                         args.samples_per_class, args.dim)[rnd_inds].view(-1, args.dim)
        neg_n = neg_f[torch.arange(0, batch_size), 1 - inds, :].view(-1,
                         args.samples_per_class, args.dim)[rnd_inds].view(-1, args.dim)

        loss_org_p1 = triplet_loss(train_proto_fg, pos_p, pos_n)
        loss_org_p2 = triplet_loss(train_proto_bg, neg_p, neg_n)
        loss_ord =  loss_org_p1 + loss_org_p2

        loss = loss_recon + loss_ord * args.lamb
        # update model
        if phase == 'Train':
            loss.backward()
            optimizer.step()

        total_losses.append(loss.item())
        recon_losses.append(loss_recon.item())
        ord_losses.append(loss_ord.item())
        ord_p1_losses.append(loss_org_p1.item())
        ord_p2_losses.append(loss_org_p2.item())

        # compute accuracy self fg
        d_a0a1 = F.pairwise_distance(pos_pooled, pooleds[:, 0])
        d_a0a2 = F.pairwise_distance(pos_pooled, pooleds[:, 1])
        sub_pix = pos_ious[:, 1] - pos_ious[:, 0]
        sub_d = d_a0a1 - d_a0a2
        mask = (torch.sign(sub_d) == torch.sign(sub_pix))
        correct_inst_fg += mask.sum().item()

        # compute accuracy self bg
        d_a0a1 = F.pairwise_distance(neg_pooled, pooleds[:, 2])
        d_a0a2 = F.pairwise_distance(neg_pooled, pooleds[:, 3])
        sub_pix_bg = neg_ious[:, 1] - neg_ious[:, 0]
        sub_d = d_a0a1 - d_a0a2
        mask = (torch.sign(sub_d) == torch.sign(sub_pix_bg))
        correct_inst_bg += mask.sum().item()

        d_a0a1 = F.pairwise_distance(train_proto_fg, pooleds[:, 0])
        d_a0a2 = F.pairwise_distance(train_proto_fg, pooleds[:, 1])
        sub_d = d_a0a1 - d_a0a2
        mask = (torch.sign(sub_d) == torch.sign(sub_pix))
        correct_proto_fg += mask.sum().item()

        d_a0a1 = F.pairwise_distance(train_proto_bg, pooleds[:, 2])
        d_a0a2 = F.pairwise_distance(train_proto_bg, pooleds[:, 3])
        sub_d = d_a0a1 - d_a0a2
        mask = (torch.sign(sub_d) == torch.sign(sub_pix_bg))
        correct_proto_bg += mask.sum().item()

        if batch_idx % args.log_interval == 0:
            if batch_idx == 0:
                print('\n')
            print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                phase, epoch, batch_idx * batch_size,
                len(data_loader.dataset),
                100. * batch_idx / len_dataloader, np.mean(total_losses)))

        break
    total_losses = np.mean(total_losses)
    acc_inst_fg = 100. * correct_inst_fg / count #len(data_loader.dataset)
    acc_proto_fg = 100. * correct_proto_fg / count #len(data_loader.dataset)
    acc_inst_bg = 100. * correct_inst_bg / count  # len(data_loader.dataset)
    acc_proto_bg = 100. * correct_proto_bg / count  # len(data_loader.dataset)

    # plot
    input = convert_image_np(
        torchvision.utils.make_grid(data[:16].cpu(), nrow=4,
                                    padding=5, pad_value=123), norm=False)
    recon_img0 = TF.normalize(recon_img, (0.1307, ), (0.3081, ))
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
                   "{} Ord P1 Loss".format(phase): np.mean(ord_p1_losses),
                   "{} Ord P2 Loss".format(phase): np.mean(ord_p2_losses),
                   '{} Ord Acc Inst FG'.format(phase): acc_inst_fg,
                   '{} Ord Acc Proto FG'.format(phase): acc_proto_fg,
                   '{} Ord Acc Inst BG'.format(phase): acc_inst_bg,
                   '{} Ord Acc Proto BG'.format(phase): acc_proto_bg,
                   "{} Images".format(phase): wandb.Image(plt)
                   }
    if phase == 'Train':
        current_lr = scheduler.get_last_lr()[0]
        log_message.update({'learning rate': current_lr})

    if upload:
        wandb.log(log_message, step=epoch)
    plt.close()

    return total_losses, acc_inst_fg, acc_proto_fg, acc_inst_bg, acc_proto_bg

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
        acc_inst_fg, acc_proto_fg, acc_inst_bg, \
        acc_proto_bg = ckpt['acc_inst_fg'], ckpt['acc_proto_fg'],\
                          ckpt['acc_inst_bg'], ckpt['acc_proto_bg']
        print(("=> loaded model checkpoint epoch {} {}\n\tacc_inst_fg {:.2f}"
               "\tacc_proto_fg {:.2f}\tacc_inst_bg {:.2f}"
               "\tacc_proto_bg {:.2f}".format(epoch, name, acc_inst_fg, acc_proto_fg,
                                                acc_inst_bg, acc_proto_bg)))
        test_accs_inst_fg, test_accs_proto_fg, test_accs_inst_bg, \
        test_accs_proto_bg = [], [], [], []
        with torch.no_grad():
            for d in range(0, 10):
                # if d == args.digit:
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
                    test_accs_inst_fg.append(test_acc_inst_fg)
                    test_accs_proto_fg.append(test_acc_proto_fg)
                    test_accs_inst_bg.append(test_acc_inst_bg)
                    test_accs_proto_bg.append(test_acc_proto_bg)
                print(d, ' test_acc_inst_fg {:.2f}\ttest_acc_proto_fg {:.2f}'
                         '\ttest_accs_inst_bg {:.2f}\ttest_acc_proto_bg {:.2f}'.format(
                      test_acc_inst_fg, test_acc_proto_fg, test_acc_inst_bg,
                      test_acc_proto_bg))
            print('mean test_acc_inst_fg {:.2f}\ttest_acc_proto_fg {:.2f}'
                     '\ttest_acc_inst_other {:.2f}\ttest_acc_proto_bg {:.2f}'.format(
                np.mean(test_accs_inst_fg), np.mean(test_accs_proto_fg), np.mean(test_accs_inst_bg),
                np.mean(test_accs_proto_bg)))
        if upload:
            wandb.run.summary.update({
            'Test {} Ord Acc Inst FG'.format(name.capitalize()): acc_inst_fg,
            'Test {} Ord Acc Proto FG'.format(name.capitalize()): acc_proto_fg,
            'Test {} Ord Acc Inst BG'.format(name.capitalize()): acc_inst_bg,
            'Test {} Ord Acc Proto BG'.format(name.capitalize()): acc_proto_bg,
            'Test {} Gen Ord Acc Inst FG'.format(name.capitalize()): np.mean(test_accs_inst_fg),
            'Test {} Gen Ord Acc Proto FG'.format(name.capitalize()): np.mean(test_accs_proto_fg),
            'Test {} Gen Ord Acc Inst BG'.format(name.capitalize()): np.mean(test_accs_inst_bg),
            'Test {} Gen Ord Acc Proto BG'.format(name.capitalize()): np.mean(test_accs_proto_bg),
            })

def extract_protos(dataloader, embedding_net):
    '''extract proto for fg & bg digit'''
    roi_pool = ROIPool((28, 28), 1.0)
    proto_fg, proto_bg = torch.zeros((len(dataloader.dataset), args.dim)).to(device), \
                         torch.zeros((len(dataloader.dataset), args.dim)).to(device)
    k = 0
    with torch.no_grad():
        for batch_idx, (data, target, _, _) in enumerate(dataloader):
            batch_size = data.shape[0]
            data, target = data.to(device), target.to(device).float()
            rois_targets = torch.cat((torch.arange(0, batch_size).unsqueeze(1).repeat(1, 2).view(-1, 1).to(device),
                                      target.view(-1, 5)[:, :4]), dim=1)  # (2bs, 5)
            roi_region = roi_pool(data, rois_targets)
            embed_org = embedding_net.get_embedding(roi_region)
            proto_fg[k:(k+batch_size)] = embed_org.view(batch_size, 2, -1)[:, 0]
            proto_bg[k:(k + batch_size)] = embed_org.view(batch_size, 2, -1)[:, 1]
            k += batch_size

    return proto_fg.mean(0), proto_bg.mean(0)


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['WANDB_NAME'] = args.savename
    wandb.init(project="selfpaced")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saveroot = os.path.join('/research/cbim/vast/tl601/results/mnist/',
                       args.savename)
    os.makedirs(saveroot, exist_ok=True)

    train_loader, test_loader = init_dataloader()

    model = AutoencoderProj(channel=1, pooling_size=args.pooling_size, dim=args.dim,
                        pooling_mode=args.pooling_mode).to(device)
    embedding_net = EmbeddingNet(dim=args.dim).to(device)
    # loaded siamese pretrained net
    model_path = os.path.join('/research/cbim/vast/tl601/results/mnist/', args.pretrained_cls)
    ckpt = torch.load(model_path)
    embedding_net.load_state_dict(ckpt['state_dict'])
    train_proto_fg, train_proto_bg = extract_protos(train_loader, embedding_net)
    test_proto_fg, test_proto_bg = extract_protos(test_loader, embedding_net)
    print('dist train fg & bg', F.pairwise_distance(train_proto_fg.unsqueeze(0),
                                                    train_proto_bg.unsqueeze(0)))
    print('dist test fg & bg', F.pairwise_distance(test_proto_fg.unsqueeze(0),
                                                    test_proto_bg.unsqueeze(0)))

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
        train_acc_fg, train_acc_bg = evaluate_siam(model, train_loader)
        test_acc_fg, test_acc_bg = evaluate_siam(model, test_loader)
        print('train acc: fg {:.4f}\tbg {:.4f}\ntest acc: fg {:.4f}\tbg {:.4f}'.format(
            train_acc_fg, train_acc_bg, test_acc_fg, test_acc_bg
        ))
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
        train_log = feed_data(model, embedding_net, data_loader=train_loader, is_train=True)
        scheduler.step()

        with torch.no_grad():
            test_loss, test_acc_inst_fg, test_acc_proto_fg, test_acc_inst_bg,\
            test_acc_proto_bg = feed_data(model, embedding_net,
                                            data_loader=test_loader, is_train=False)

            # save model
            if test_acc_proto_fg > best_acc:
                save_model = True
                best_acc = test_acc_proto_fg
                no_improve_epoch = 0
            else:
                save_model = False
                no_improve_epoch += 1
            if save_model:
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'acc_inst_fg': test_acc_inst_fg,
                    'acc_proto_fg': test_acc_proto_fg,
                    'acc_inst_bg': test_acc_inst_bg,
                    'acc_proto_bg': test_acc_proto_bg
                }, os.path.join(saveroot,'best.pth.tar'))

            # break training
            if no_improve_epoch > args.patience:
                print('stop training...')
                break

        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'acc_inst_fg': test_acc_inst_fg,
            'acc_proto_fg': test_acc_proto_fg,
            'acc_inst_bg': test_acc_inst_bg,
            'acc_proto_bg': test_acc_proto_bg
        }, os.path.join(saveroot, 'last.pth.tar'))

    #evaluate()





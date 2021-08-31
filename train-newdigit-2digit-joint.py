"""
Using pretrained embedding network and agent, only finetune agent.
Created on 4/22/2021 10:36 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import torchvision
import os, torch, wandb
from opts import parser
import numpy as np
import torch.nn as nn
from itertools import product
from torch.distributions import Categorical
import torch.nn.functional as F
from RL_entropy import get_policy_loss
from util.utils import box_iou
from util.augmentations import Compose
from util.utils import generate_boxes
from util.actions_tree import Actor
import torchvision.transforms.functional as TF
from util.data_aug import Resize,RandomTranslate, RandomHorizontalFlip, RandomShear
from datasets.clutter_mnist_2digit_newdigit import MNIST_CoLoc
from datasets.clutter_mnist_support import MNIST_Supp
from models.mnist_scale_model import Net_ae_proj, Agent_ae
from models.mnist_ae_ord import AutoencoderProj
from util.utils import convert_image_np
import matplotlib.pyplot as plt

DEBUG = True

def init_dataloader():
    kwargs = {'num_workers': 8, 'pin_memory': True}

    # train_transform = Compose([RandomHorizontalFlip(),
    #                            # RandomShear(),
    #                            RandomTranslate(),
    #                            Resize(84)])
    trainset = MNIST_CoLoc(root='.', train=True, digit=args.digit,
                           support_size=args.support_size, sample_size=args.sample_size,
                           datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                           clutter=1)
    testset = MNIST_CoLoc(root='.', train=False, digit=args.digit,
                          datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                          clutter=1)
    anchors = generate_boxes(base_size=4,
                             feat_height=21, feat_width=21, img_size=84,
                             feat_stride=4,
                             ratios=[1.0],
                             min_box_side=28,
                             scales=np.array(range(7, 20)))
    print('number of anchors for 84*84 image ', anchors.shape[0])
    suppset = MNIST_Supp(supp_data=trainset.supp_data,
                         supp_targets=trainset.supp_targets,
                         anchors=anchors)

    supp_loader = torch.utils.data.DataLoader(
        suppset, batch_size=512, shuffle=True, **kwargs)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    print('total train image: ', len(train_loader.dataset), ' test image: ',
          len(test_loader.dataset))
    return train_loader, test_loader, supp_loader

def init_model():
    embed_net = AutoencoderProj(pooling_size=args.pooling_size,
              pooling_mode=args.pooling_mode).to(device)
    agent = Agent_ae(rnn=args.rnn, poolsize=args.pooling_size,
                     num_class=args.num_act,
                     hidden_size=args.hidden_size).to(device)

    # if load pretrained encoder
    if args.pretrained != '':
        ckpt = torch.load(os.path.join('/research/cbim/vast/tl601/results/'
                                       'mnist/', args.pretrained))
        if 'state_dict' in ckpt.keys():
            mnist_pretrained_dict = ckpt['state_dict']
        else:
            mnist_pretrained_dict = ckpt
        embeddingnet_dict = embed_net.state_dict()

        filtered_dict = {}
        for k, v in mnist_pretrained_dict.items():
            if k in embeddingnet_dict:
                print('load ', k)
                filtered_dict[k] = v
        embeddingnet_dict.update(filtered_dict)
        embed_net.load_state_dict(embeddingnet_dict)
        if 'state_dict' in ckpt.keys():
            print('loaded from {}\nckpt epoch {} acc {:.2f}%'.format(args.pretrained,
                                                                     ckpt['epoch'], ckpt['acc_inst']))
        else:
            print('loaded pretained MNIST encoder ', args.pretrained)

    # load pretrained agent on digit 4
    agent_path = os.path.join('/research/cbim/vast/tl601/results/selfpaced/ordinal',
                              args.pretrained_agent)
    print('loading ckpt from ', agent_path)
    ckpt = torch.load(agent_path)
    state_dict = ckpt['state_dict']
    epoch = ckpt['epoch']
    best_loss = ckpt['acc']
    agent.load_state_dict(state_dict)
    print(("=> loaded checkpoint epoch {} {}".format(epoch, best_loss)))

    # only freeze encoder
    #assert (args.freeze == 0)
    if args.freeze == 1:
        for name, value in embed_net.named_parameters():
            value.requires_grad = False
        embed_net.eval()

    return embed_net, agent

def feed_data_embed(embed_net, supp_loader, is_train, upload=True):
    if is_train:
        phase = 'Train'
        embed_net.train()
    else:
        phase = 'Test'
        embed_net.eval()

    total_losses, recon_losses, ord_losses = [], [], []
    correct_inst, correct_proto = 0, 0
    len_dataloader = len(supp_loader)

    for batch_idx, (data, target, ti, tj, ioui, iouj) in enumerate(supp_loader):
        batch_size = data.shape[0]
        data, target, ti, tj, ioui, iouj = data.to(device), \
                                          target.to(device).float(), \
                                          ti.to(device), tj.to(device), \
                                          ioui.to(device), iouj.to(device)
        if is_train:
            optimizer.zero_grad()
        # rois
        rois_0 = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                            target), dim=1)
        rois_i = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                            ti), dim=1)
        rois_j = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                            tj), dim=1)

        recon_img0, pooled_0 = embed_net(data, rois_0)  # (bs, 1, 84, 84), (bs, 1600)
        recon_imgi, pooled_i = embed_net(data, rois_i)
        recon_imgj, pooled_j = embed_net(data, rois_j)

        loss_recon = criterion(recon_img0, data)

        # compute loss order
        iou = torch.stack((ioui, iouj), 1)  # (bs, 2)
        f = torch.stack((pooled_i, pooled_j)).permute(1, 0, 2)  # (bs, 2, dim)

        inds = torch.max(iou, 1).indices
        p = f[torch.arange(0, batch_size), inds, :]
        n = f[torch.arange(0, batch_size), 1 - inds, :]
        proto = torch.mean(pooled_0, dim=0).view(1, -1).repeat(batch_size, 1)
        #loss_ord = triplet_loss(proto, p, n)
        loss_ord = triplet_loss(pooled_0, p, n)

        loss = loss_recon + loss_ord * 0.1
        # update model
        if phase == 'Train':
            loss.backward()
            optimizer.step()

        total_losses.append(loss.item())
        recon_losses.append(loss_recon.item())
        ord_losses.append(loss_ord.item())

        # compute accuracy
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

        if batch_idx % args.log_interval == 0:
            if batch_idx == 0:
                print('\n')
            print('{} Embed Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                phase, epoch, batch_idx * batch_size,
                len(supp_loader.dataset),
                100. * batch_idx / len_dataloader, np.mean(total_losses)))

        #break

    total_losses = np.mean(total_losses)
    acc_inst = 100. * correct_inst / len(supp_loader.dataset)
    acc_proto = 100. * correct_proto / len(supp_loader.dataset)

    log_message = {"{} Total Embed Loss".format(phase): total_losses,
                   "{} Recon Loss".format(phase): np.mean(recon_losses),
                   "{} Ord Loss".format(phase): np.mean(ord_losses),
                   '{} Ord Acc Inst'.format(phase): acc_inst,
                   '{} Ord Acc Proto'.format(phase): acc_proto,
                   }


    if phase == 'Train':
        '''update embedding'''
        embed_net.eval()
        rois_t0 = torch.cat((torch.arange(0, supp_data.shape[0]).float().view(-1, 1),
                             supp_target), dim=1).to(device)
        supp_embed, supp_top_feat = embed_net(supp_data, rois_t0)  # (support_size, 512)
        if use_top_feat:
            proto_feat = torch.mean(supp_top_feat, dim=0)  # (512,)
        else:
            # (64*poolsize*poolsize,)
            proto_feat = torch.mean(supp_embed.view(supp_embed.shape[0], -1), dim=0)

        '''plot'''
        input = convert_image_np(
            torchvision.utils.make_grid(data[:16].cpu(), nrow=4,
                                        padding=5, pad_value=123), norm=False)
        recon_img0 = TF.normalize(recon_img0, (0.1307,), (0.3081,))
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

        current_lr = scheduler.get_last_lr()[0]
        log_message.update({'learning rate': current_lr,
                            "{} Images".format(phase): wandb.Image(plt)})

    wandb.log(log_message, step=epoch)
    if phase == 'Train':
        plt.close()
        return total_losses, acc_proto, proto_feat
    else:
        return  total_losses, acc_proto

def feed_data(embed_net, agent, data_loader, is_train, upload=True):
    embed_net.eval()
    if is_train:
        phase = 'Train'
        agent.train()
        #embed_net.train()
    else:
        phase = 'Test'
        agent.eval()
        #embed_net.eval()

    pol_losses, ent_losses, rewards = [], [], []
    correct = 0
    len_dataloader = len(data_loader)
    for batch_idx, (data, target) in enumerate(data_loader):
        batch_size = data.shape[0]
        data, target = data.to(device),  target.float().squeeze()
        optimizer_ag.zero_grad()

        '''trian agent'''
        actor = Actor(data, min_box_side=28)
        rewards_all = torch.zeros((batch_size, args.seq_len + 1)).to(device)
        action_seq = torch.IntTensor(batch_size, args.seq_len).to(device)
        logits_seq, boxes = [], []
        all_embed = []
        if args.hidden_size:
            h_t = torch.zeros(
                batch_size,
                args.hidden_size,
                dtype=torch.float,
                device=device,
                requires_grad=True,
            )

        pred_boxes = torch.tensor([[0, 0, 83, 83]]).float().repeat(batch_size, 1)
        rois = torch.cat((torch.arange(0, batch_size).float().view(-1, 1),
                          pred_boxes), dim=1).to(device)
        # 1st forward encoder
        with torch.no_grad():
            embed, top_feat = embed_net.get_embedding(data, rois)
        # compute reward for the initial step
        if use_top_feat:
            rewards_all[:, 0] = F.pairwise_distance(proto_feat, top_feat)
        else:
            rewards_all[:, 0] = F.pairwise_distance(proto_feat, embed.view(embed.shape[0], -1))

        for t in range(args.seq_len):
            if args.hidden_size:
                h_t, logits, actions = agent(embed, h_t_prev=h_t)
            else:
                logits, actions = agent(embed)
            state, pred_boxes = actor.takeaction(actions)
            rois = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                              pred_boxes.to(device)), dim=1)
            with torch.no_grad():
                embed, top_feat = embed_net.get_embedding(data, rois)
            if use_top_feat:
                all_embed.append(top_feat)
            else:
                all_embed.append(embed.view(embed.shape[0], -1))
            action_seq[:, t] = actions.view(1, -1)
            logits_seq.append(logits)
            boxes.append(pred_boxes)

        rewards_trns = F.pairwise_distance(proto_feat, torch.cat(all_embed))
        rewards_all[:, 1:] = rewards_trns.view(-1, batch_size).permute(1, 0)

        rewards_deltas = -rewards_all[:, 1:] - (-rewards_all[:, :-1])

        # Get the policy loss
        coord = product(range(action_seq.size(0)), range(action_seq.size(1)))
        coo_actions = [[k, m, action_seq[k, m]] for k, m in coord]

        logits_seq = torch.stack(tuple(logits_seq), dim=0)
        logits_seq = logits_seq.permute(1, 0, 2).to(device)  # (bs, T, dim)

        # TODO, deal with stop action, modify rewards
        q, score, critic_loss, adv = get_policy_loss(rewards_deltas, batch_size, gamma=args.gamma,
                                                     logits_seq=logits_seq, seq_len=args.seq_len,
                                                     coo_actions=coo_actions, values=None)

        m = Categorical(logits=logits_seq)
        loss_ent = m.entropy()  # (bs, T)

        loss = (q - loss_ent * args.lamb_ent).sum()
        if phase == 'Train':
            loss.backward()
            optimizer_ag.step()

        pol_losses.append(q.sum().item())
        ent_losses.append(loss_ent.sum().item())
        rewards.append(torch.mean(rewards_deltas).item())

        # calculate accuracy
        iou = torch.diagonal(box_iou(pred_boxes.float(),
                                     target), 0)
        correct += (iou >= 0.5).sum()

        if DEBUG and batch_idx % args.log_interval == 0:
            print('action_seq ', action_seq[:5])
            print('rewards_all ', -rewards_all[:5])
            print('rewards_deltas ', rewards_deltas[:5])
            print('adv ', adv[:5])
            print('pred_boxes ', pred_boxes[:5])
            print('target_boxes ', target[:5].int())

        if batch_idx % args.log_interval == 0:
            if batch_idx == 0:
                print('\n')
            print('{} Epoch: {} [{}/{} ({:.0f}%)]\tReward: {:.4f}'.format(
                phase, epoch, batch_idx * batch_size,
                len(data_loader.dataset),
                100. * batch_idx / len_dataloader, np.mean(rewards)))

        #break

    total_pol_loss = np.mean(pol_losses)
    acc = 100. * correct / len(data_loader.dataset)
    log_message = {"{} Pol Loss".format(phase): total_pol_loss,
                   '{} Rewards'.format(phase): np.mean(rewards),
                   '{} Loc Acc'.format(phase): acc,
                   '{} Ent Loss'.format(phase): np.mean(ent_losses)
                   }

    if phase == 'Train':
        current_lr = scheduler_ag.get_last_lr()[0]
        log_message.update({'learning rate': current_lr})
    if upload:
        wandb.log(log_message, step=epoch)

    return total_pol_loss, acc

def evaluate(upload=True):
    net_path = os.path.join(saveroot, 'best_net.pth.tar')
    agent_path = os.path.join(saveroot, 'best_agent.pth.tar')
    # load net
    print('loading net ckpt from ', net_path)
    ckpt = torch.load(net_path)
    state_dict = ckpt['state_dict']
    epoch = ckpt['epoch']
    best_acc_proto = ckpt['acc_proto']
    embed_net.load_state_dict(state_dict)
    print(("=> loaded net checkpoint epoch {} {}".format(epoch, best_acc_proto)))
    # load agent
    print('loading net ckpt from ', agent_path)
    ckpt = torch.load(agent_path)
    state_dict = ckpt['state_dict']
    epoch = ckpt['epoch']
    best_acc = ckpt['acc']
    agent.load_state_dict(state_dict)
    print(("=> loaded agent checkpoint epoch {} {}".format(epoch, best_acc)))
    if upload:
        wandb.run.summary.update({'Test Best Loc Acc': best_acc,
                                  'Test Best Ord Acc': best_acc_proto})

if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ['WANDB_NAME'] = args.savename
    wandb.init(project="selfpaced")
    saveroot = os.path.join('/research/cbim/vast/tl601/results/selfpaced/ordinal',
                            args.savename)
    os.makedirs(saveroot, exist_ok=True)

    train_loader, test_loader, supp_loader = init_dataloader()

    embed_net, agent = init_model()

    wandb.watch(agent, log_freq=10)
    wandb.watch(embed_net, log_freq=10)
    wandb.config.update(args)

    # flag of using projection head embedding for ordinal+ae or not for ae
    use_top_feat = False
    if 'ordinal' in args.pretrained_agent:
        use_top_feat = True

    # compute embedding
    supp_data, supp_target = train_loader.dataset.supp_data.to(device), \
                             train_loader.dataset.supp_targets.float()
    rois_t0 = torch.cat((torch.arange(0, supp_data.shape[0]).float().view(-1, 1),
                         supp_target), dim=1).to(device)
    with torch.no_grad():
        supp_embed, supp_top_feat = embed_net(supp_data, rois_t0) # (support_size, 512)
    if use_top_feat:
        proto_feat = torch.mean(supp_top_feat, dim=0) # (512,)
    else:
        # (64*poolsize*poolsize,)
        proto_feat = torch.mean(supp_embed.view(supp_embed.shape[0], -1), dim=0)

    criterion = nn.MSELoss()  # mean square error loss
    triplet_loss = nn.TripletMarginLoss(margin=args.margin, p=2)
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(embed_net.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(embed_net.parameters(), lr=args.lr)
    if args.optimizer_ag == 'SGD':
        optimizer_ag = torch.optim.SGD(agent.parameters(), lr=args.lr_ag, momentum=0.9)
    elif args.optimizer_ag == 'Adam':
        optimizer_ag = torch.optim.Adam(agent.parameters(), lr=args.lr_ag)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=args.steps, gamma=0.1)
    scheduler_ag = torch.optim.lr_scheduler.StepLR(optimizer_ag,
                                                   step_size=args.step_ag, gamma=0.1)

    best_acc = 0.
    save_model = False
    no_improve_epoch = 0
    for epoch in range(args.epochs):
        train_log = feed_data(embed_net, agent, data_loader=train_loader, is_train=True)
        if epoch < 3:
            train_log, _, proto_feat = feed_data_embed(embed_net,
                                                supp_loader=supp_loader, is_train=True)
        scheduler_ag.step()
        scheduler.step()

        with torch.no_grad():
            test_loss, test_acc = feed_data(embed_net, agent,
                                        data_loader=test_loader, is_train=False)
            if epoch < 3:
                test_embed_loss, test_acc_proto = feed_data_embed(embed_net,
                                                supp_loader=supp_loader, is_train=False)

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
                    'state_dict': agent.state_dict(),
                    'acc': best_acc
                }, os.path.join(saveroot, 'best_agent.pth.tar'))
                torch.save({
                    'epoch': epoch,
                    'state_dict': embed_net.state_dict(),
                    'acc_proto': test_acc_proto
                }, os.path.join(saveroot, 'best_net.pth.tar'))

            # break training
            if no_improve_epoch > args.patience:
                print('stop training...')
                break

    torch.save({
        'epoch': epoch,
        'state_dict': agent.state_dict(),
        'acc': test_acc
    }, os.path.join(saveroot, 'last_agent.pth.tar'))
    torch.save({
        'epoch': epoch,
        'state_dict': embed_net.state_dict(),
        'acc_proto': test_acc_proto
    }, os.path.join(saveroot, 'last_net.pth.tar'))
    # evaluate gen acc
    evaluate()





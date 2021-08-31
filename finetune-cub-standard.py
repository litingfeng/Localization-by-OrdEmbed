"""
adap agent from training with loose annotation to stancard
Created on 5/25/2021 2:07 AM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import os, wandb
import torch, copy

import torch.nn as nn
import torch.nn.functional as F
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from transform import MyBoxScaleTransform
from util.data_aug import *
from util.bbox_util import *
from util import joint_transforms as t
from torch.distributions import Categorical
from util.actions_tree import Actor
from util.utils import box_iou
from RL_entropy import get_policy_loss
from opts import parser
from datasets.cub_fewshot_std import CUB_fewshot, init_sampler
from models.mnist_scale_model import Agent_ae
from models.cub_scale_model import Net
from constants import IMAGENET_MEAN, IMAGENET_STD

DEBUG = False

def init_dataloader():
    transform = t.Compose([
        t.ConvertFromPIL(),
        t.ToPercentCoords(),
        t.Resize(args.img_size),
        t.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        t.ToTensor()  # no change to (0, 1)
    ])
    kwargs = {'num_workers': 8, 'pin_memory': True}
    trainset = CUB_fewshot('/research/cbim/vast/tl601/Dataset/CUB_200_2011',
                           mode='val', img_size=args.img_size, bg_name=args.bg_name,
                           support_size=args.support_size,
                           transform=transform)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    print('total train image: ', len(train_loader.dataset))
    return train_loader

def init_model():
    # small model
    # net = Net_ae_proj(channel=3, pooling_size=args.pooling_size,
    #           pooling_mode=args.pooling_mode).to(device)
    net = Net(pooling_size=args.pooling_size, pretrained=True, dim=args.dim,
                      pooling_mode=args.pooling_mode).to(device)
    agent = Agent_ae(rnn=args.rnn, dim_ag=512, poolsize=args.pooling_size,
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
        embeddingnet_dict = net.state_dict()

        filtered_dict = {}
        for k, v in mnist_pretrained_dict.items():
            if k in embeddingnet_dict:
                filtered_dict[k] = v
        embeddingnet_dict.update(filtered_dict)
        net.load_state_dict(embeddingnet_dict)
        if 'state_dict' in ckpt.keys():
            print('loaded from {}\nckpt epoch {} acc inst {:.2f}%'.format(args.pretrained,
                                                    ckpt['epoch'], ckpt['acc_inst']))
        else:
            print('loaded pretained MNIST encoder ', args.pretrained)

    if args.pretrained_agent != '':
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

    if args.freeze == 1:
        for name, value in net.named_parameters():
            value.requires_grad = False
        net.eval()

    return net, agent

def feed_data(model, agent, data_loader, is_train, upload=True):
    if is_train:
        phase = 'Train'
        agent.train()
    else:
        phase = 'Test'
        agent.eval()

    pol_losses, ent_losses, rewards = [], [], []
    correct = 0
    len_dataloader = len(data_loader)
    for batch_idx, (data, target) in enumerate(data_loader):
        batch_size = data.shape[0]
        data, target = data.to(device), target.float()
        optimizer_ag.zero_grad()

        actor = Actor(data, min_box_side=args.min_box_side)
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

        pred_boxes = torch.tensor([[0, 0, args.img_size, args.img_size]]).float().repeat(batch_size, 1)
        rois = torch.cat((torch.arange(0, batch_size).float().view(-1, 1),
                          pred_boxes), dim=1).to(device)
        # 1st forward encoder
        with torch.no_grad():
            embed, top_feat, _ = embed_net(data, rois)

        rewards_all[:, 0] = F.pairwise_distance(proto_feat, top_feat)

        for t in range(args.seq_len):
            if args.hidden_size:
                h_t, logits, actions = agent(embed, h_t_prev=h_t)
            else:
                logits, actions = agent(embed)
            state, pred_boxes = actor.takeaction(actions)
            rois = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                              pred_boxes.to(device)), dim=1)
            with torch.no_grad():
                embed, top_feat = embed_net.get_roi_embedding(rois)

            all_embed.append(top_feat)

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

        loss = (q * args.lamb_pol - loss_ent * args.lamb_ent).sum()
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
            print('rewards_all ', rewards_all[:5])
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
    agent_path = os.path.join(saveroot, 'best.pth.tar')
    # load agent
    print('loading net ckpt from ', agent_path)
    ckpt = torch.load(agent_path)
    state_dict = ckpt['state_dict']
    epoch = ckpt['epoch']
    best_acc = ckpt['acc']
    agent.load_state_dict(state_dict)
    print(("=> loaded agent checkpoint epoch {} {}".format(epoch, best_acc)))
    if upload:
        wandb.run.summary.update({'Test Best Loc Acc': best_acc})
    test_loss, test_acc = feed_data(embed_net, agent,
                                    data_loader=train_loader, is_train=False,
                                    upload=False)
    print('after finetuning train_acc ', test_acc)

if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ['WANDB_NAME'] = args.savename
    wandb.init(project="selfpaced")
    saveroot = os.path.join('/research/cbim/vast/tl601/results/selfpaced/ordinal',
                            args.savename)
    os.makedirs(saveroot, exist_ok=True)

    train_loader = init_dataloader()

    embed_net, agent = init_model()

    wandb.watch(agent, log_freq=10)
    wandb.config.update(args)

    # compute embedding
    supp_data, supp_target = train_loader.dataset.supp_data.to(device), \
                             train_loader.dataset.supp_targets.float()
    rois_t0 = torch.cat((torch.arange(0, supp_data.shape[0]).float().view(-1, 1),
                         supp_target), dim=1).to(device)
    with torch.no_grad():
        supp_embed, supp_top_feat, _ = embed_net(supp_data, rois_t0) # (support_size, 512)
    proto_feat = torch.mean(supp_top_feat, dim=0) # (512,)

    if args.optimizer_ag == 'SGD':
        optimizer_ag = torch.optim.SGD(agent.parameters(), lr=args.lr_ag, momentum=0.9)
    elif args.optimizer_ag == 'Adam':
        optimizer_ag = torch.optim.Adam(agent.parameters(), lr=args.lr_ag)
    scheduler_ag = torch.optim.lr_scheduler.StepLR(optimizer_ag,
                                                   step_size=args.step_ag, gamma=0.1)

    epoch = 0
    with torch.no_grad():
        train_log = feed_data(embed_net, agent, data_loader=train_loader, is_train=False)
        print('before finetune... acc ', train_log[1])
        #exit()

    best_acc = 0.
    save_model = False
    no_improve_epoch = 0
    for epoch in range(args.epochs):
        train_log = feed_data(embed_net, agent, data_loader=train_loader, is_train=True)
        scheduler_ag.step()

        with torch.no_grad():
            test_loss, test_acc = feed_data(embed_net, agent,
                                        data_loader=train_loader, is_train=False)

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
                'state_dict': agent.state_dict(),
                'acc': best_acc
            }, os.path.join(saveroot, 'best.pth.tar'))
        # break training
        if no_improve_epoch > args.patience:
            print('stop training...')
            break

        torch.save({
            'epoch': epoch,
            'state_dict': agent.state_dict(),
        }, os.path.join(saveroot, 'last.pth.tar'))
    evaluate()

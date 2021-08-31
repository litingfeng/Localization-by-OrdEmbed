"""
omniglot dataset
Use pretrained ordinal embedding to train RL in phase A. Support defined reward
Created on 3/12/2021 7:18 PM

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
from datasets.cub_fewshot_rl import CUB_fewshot, init_sampler
from models.mnist_scale_model import Agent, Net_ae, Net_ae_proj, Agent_ae, Agent_cub
from models.cub_scale_model import Net, Agent
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
    trainset = CUB_fewshot('/research/cbim/vast/tl601/Dataset/CUB_200_2011',
                           mode='base', img_size=args.img_size, loose=args.loose,
                           transform=transform)
    testset = CUB_fewshot('/research/cbim/vast/tl601/Dataset/CUB_200_2011',
                          mode='val', img_size=args.img_size,
                          transform=transform)
    kwargs = {'num_workers': 8, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    print('total train image: ', len(train_loader.dataset), ' test image: ',
          len(test_loader.dataset))
    return train_loader, test_loader

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
                print('load ', k)
                filtered_dict[k] = v
        embeddingnet_dict.update(filtered_dict)
        net.load_state_dict(embeddingnet_dict)
        if 'state_dict' in ckpt.keys():
            print('loaded from {}\nckpt epoch {} acc inst {:.2f}%'.format(args.pretrained,
                                                    ckpt['epoch'], ckpt['acc_inst']))
        else:
            print('loaded pretained MNIST encoder ', args.pretrained)

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
    correct, count = 0, 0
    len_dataloader = len(data_loader)
    for batch_idx, (data, target, _) in enumerate(data_loader):
        batch_size = data.shape[0]
        data, target = data.to(device),  target.float()

        '''split to support set and target set '''
        num = batch_size % args.samples_per_class
        if num != 0:
            data, target = data[:-num], target[:-num]
            batch_size = data.shape[0]

        num_group = batch_size // args.samples_per_class
        rnd_inds = np.random.permutation(num_group)

        optimizer_ag.zero_grad()

        # rois
        rois_t0 = torch.cat((torch.arange(0, batch_size).float().view(-1, 1),
                            target[:, :4]), dim=1).to(device)

        actor = Actor(data, min_box_side=args.min_box_side) # img 84, min 16; img 128, min 24

        rewards_all = torch.zeros((batch_size, args.seq_len+1)).to(device)
        action_seq = torch.IntTensor(batch_size, args.seq_len).to(device)
        logits_seq, boxes = [], []
        all_embed = []
        if args.rnn == 1:
            h_t = torch.zeros(
                batch_size,
                args.hidden_size,
                dtype=torch.float,
                device=device,
                requires_grad=True,
            )

        pred_boxes = torch.tensor([[0, 0,
                    args.img_size-1, args.img_size-1]]).float().repeat(batch_size, 1).to(device)
        rois = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                          pred_boxes), dim=1)
        # 1st forward classifier
        with torch.no_grad():
            embed_t, top_feat_t, _ = model(data, rois_t0)
            embed, top_feat, _ = model(data, rois)
        if 'proto' in args.train_mode:  #
            proto = top_feat_t.view(-1, args.samples_per_class, args.dim).mean(1)  # (num_group, 512)
            if args.train_mode == 'shuffle_proto':  # shuffle proto or top top_feat?
                proto = proto[rnd_inds]
                # rewards_all[:, 0] = F.pairwise_distance(proto, top_feat.view(-1,
                #                        args.samples_per_class, 512)[rnd_inds].view(-1,512))
            proto = proto.unsqueeze(1).repeat(1, args.samples_per_class, 1).view(-1, args.dim)
            rewards_all[:, 0] = F.pairwise_distance(proto, top_feat)
        else:
            if args.train_mode == 'shuffle_self':
                top_feat_t = top_feat_t.view(-1, args.samples_per_class, args.dim)[rnd_inds].view(-1, args.dim)
            rewards_all[:, 0] = F.pairwise_distance(top_feat_t, top_feat)

        for t in range(args.seq_len):
            if args.rnn == 1:
                h_t, logits, actions = agent(embed, h_t_prev=h_t)
            else:
                logits, actions = agent(embed)
            state, pred_boxes = actor.takeaction(actions)
            rois = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                              pred_boxes.to(device)), dim=1)
            with torch.no_grad():
                embed, top_feat = model.get_roi_embedding(rois)
            if 'proto' in args.train_mode:  #
                rewards_all[:, t + 1] = F.pairwise_distance(proto, top_feat)
            else:
                rewards_all[:, t + 1] = F.pairwise_distance(top_feat_t, top_feat)

            action_seq[:, t] = actions.view(1, -1)
            logits_seq.append(logits)
            boxes.append(pred_boxes)

        if args.sparse == 0:
            # incremental reward
            rewards_deltas = -rewards_all[:, 1:] - (-rewards_all[:, :-1])
            if args.sign == 1:
                rewards_deltas = torch.sign(rewards_deltas)
        else:
            # sparse reward
            rewards_deltas = - (rewards_all[:, -1] - rewards_all[:, 0])
            rewards_deltas = rewards_deltas.unsqueeze(1).repeat(1, args.seq_len)
            if args.sign == 1:
                rewards_deltas = torch.sign(rewards_deltas)

        # Get the policy loss
        coord = product(range(action_seq.size(0)), range(action_seq.size(1)))
        coo_actions = [[k, m, action_seq[k, m]] for k, m in coord]

        logits_seq = torch.stack(tuple(logits_seq), dim=0)
        logits_seq = logits_seq.permute(1, 0, 2).to(device) # (bs, T, dim)

        values = None
        #TODO, deal with stop action, modify rewards
        q, score, critic_loss, adv = get_policy_loss(rewards_deltas, batch_size, gamma=args.gamma,
                                                     logits_seq=logits_seq, seq_len=args.seq_len,
                                                     coo_actions=coo_actions, values=values)
        m = Categorical(logits=logits_seq)
        loss_ent = m.entropy() # (bs, T)

        loss = (q - loss_ent * args.lamb_ent).sum()

        if phase == 'Train':
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(agent.parameters(), 10)
            optimizer_ag.step()

        # calculate accuracy
        iou = torch.diagonal(box_iou(pred_boxes.float(),
                                     target[:,:4]), 0)
        correct += (iou >= 0.5).sum()

        pol_losses.append(q.sum().item())
        ent_losses.append(loss_ent.sum().item())
        rewards.append(torch.mean(rewards_deltas).item())

        if DEBUG:
            print('action_seq ', action_seq[:5])
            print('rewards_all ', -rewards_all[:5])
            print('rewards_deltas ', rewards_deltas[:5])
            print('adv ', adv[:5])
            print('pred_boxes ', pred_boxes[:5])
            print('target_boxes ', target[:5].int())

        count += batch_size
        if batch_idx % args.log_interval == 0:
            if batch_idx == 0:
                print('\n')
            print('{} Epoch: {} [{}/{} ({:.0f}%)]\tReward: {:.4f}'.format(
                phase, epoch, batch_idx * batch_size,
                len(data_loader.dataset),
                100. * batch_idx / len_dataloader, np.mean(rewards)))

        #break

    total_pol_loss = np.mean(pol_losses)
    acc = 100. * correct / count
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

def evaluate_acc(upload=True):
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

if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ['WANDB_NAME'] = args.savename
    wandb.init(project="selfpaced")
    saveroot = os.path.join('/research/cbim/vast/tl601/results/selfpaced/ordinal',
                            args.savename)
    os.makedirs(saveroot, exist_ok=True)

    train_loader, test_loader = init_dataloader()

    net, agent = init_model()
    #wandb.watch(net, log_freq=10)
    wandb.watch(agent, log_freq=10)
    wandb.config.update(args)

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    if args.optimizer_ag == 'SGD':
        optimizer_ag = torch.optim.SGD(agent.parameters(), lr=args.lr_ag, momentum=0.9)
    elif args.optimizer_ag == 'Adam':
        optimizer_ag = torch.optim.Adam(agent.parameters(), lr=args.lr_ag)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=args.steps, gamma=0.1)
    scheduler_ag = torch.optim.lr_scheduler.StepLR(optimizer_ag,
                                                   step_size=args.step_ag, gamma=0.1)

    if args.evaluate == 1:
        evaluate_acc(False)
        exit()

    best_acc = 0.
    save_model = False
    no_improve_epoch = 0
    for epoch in range(args.epochs):
        train_log = feed_data(net, agent, data_loader=train_loader, is_train=True)
        scheduler_ag.step()

        with torch.no_grad():
            test_loss, test_acc = feed_data(net, agent,  data_loader=test_loader, is_train=False)

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
            'acc': test_acc
        }, os.path.join(saveroot, 'last.pth.tar'))
    # evaluate gen acc
    # DEBUG = False
    evaluate_acc()
"""
https://github.com/wyharveychen/CloserLookFewShot/blob/e03aca8a2d01c9b5861a5a816cd5d3fdfc47cd45/methods/maml.py
No support set
Created on 5/14/2021 2:53 PM

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
from datasets.clutter_omg_fewshot_rl import OmniglotDataset, init_sampler
from models.cub_maml_model import Net_ae_proj, Agent_ae
from constants import IMAGENET_MEAN, IMAGENET_STD

DEBUG = False
def init_dataloader():
    trainset = OmniglotDataset(mode='train',
                               root='/research/cbim/vast/tl601/Dataset/Omniglot')
    testset = OmniglotDataset(mode='val',
                              root='/research/cbim/vast/tl601/Dataset/Omniglot')
    train_batchsampler = init_sampler(args.classes_per_set, args.samples_per_class * 2,
                                      trainset.y[:, -1], args.iterations, 'train')
    test_batchsampler = init_sampler(args.classes_per_set, args.samples_per_class*2,
                                     testset.y[:, -1],
                                     args.iterations_test, 'val')
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_sampler=train_batchsampler)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_sampler=test_batchsampler)
    print('total train image: ', len(train_loader.dataset), ' test image: ',
          len(test_loader.dataset))
    return train_loader, test_loader

def init_model():
    # using ae+ordinal+vgg
    # net = Net(pooling_size=args.pooling_size, pretrained=True,
    #             pooling_mode=args.pooling_mode).to(device)
    net = Net_ae_proj(pooling_size=args.pooling_size,
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

def localize(model, agent, data, target, task_step=0):
    batch_size = data.shape[0]
    rois_t0 = torch.cat((torch.arange(0, batch_size).float().view(-1, 1),
                         target[:, :4]), dim=1).to(device)
    actor = Actor(data.cpu(), min_box_side=args.min_box_side)  # img 84, min 16; img 128, min 24
    rewards_all = torch.zeros((batch_size, args.seq_len + 1)).to(device)
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
                                83, 83]]).float().repeat(batch_size, 1).to(device)
    rois = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                      pred_boxes), dim=1)
    with torch.no_grad():
        embed_t, top_feat_t = model(data, rois_t0)
        embed, top_feat = model.get_roi_embedding(rois)
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
        all_embed.append(top_feat)
        rewards_all[:, t+1] = F.pairwise_distance(top_feat_t, top_feat)
        action_seq[:, t] = actions.view(1, -1)
        logits_seq.append(logits)
        boxes.append(pred_boxes)

    rewards_deltas = -rewards_all[:, 1:] - (-rewards_all[:, :-1])

    # Get the policy loss
    coord = product(range(action_seq.size(0)), range(action_seq.size(1)))
    coo_actions = [[k, m, action_seq[k, m]] for k, m in coord]

    logits_seq = torch.stack(tuple(logits_seq), dim=0)
    logits_seq = logits_seq.permute(1, 0, 2).to(device)  # (bs, T, dim)

    values = None
    q, score, critic_loss, adv = get_policy_loss(rewards_deltas, batch_size, gamma=args.gamma,
                                                 logits_seq=logits_seq, seq_len=args.seq_len,
                                                 coo_actions=coo_actions, values=values)

    m = Categorical(logits=logits_seq)
    loss_ent = m.entropy()  # (bs, T)

    loss = (q - loss_ent * args.lamb_ent).sum()

    iou = torch.diagonal(box_iou(pred_boxes.float(),
                                 target[:, :4]), 0)
    if DEBUG and task_step == args.task_update_num - 1:
        print('action_seq ', action_seq[:5])
        print('rewards_all ', -rewards_all[:5])
        print('rewards_deltas ', rewards_deltas[:5])
        print('adv ', adv[:5])
        print('pred_boxes ', pred_boxes[:5])
        print('target_boxes ', target[:5].int())

    return loss, q, loss_ent, rewards_deltas, iou

def feed_data(model, agent, data_loader, is_train= True, upload=True):
    if is_train:
        phase = 'Train'
    else:
        phase = 'Test'
    agent.train()

    loss_all, pol_losses, ent_losses, rewards = [], [], [], []
    task_count = 0
    corrects = [0 for _ in range(2)] # correct before and after update
    len_dataloader = len(data_loader)
    for batch_idx, (data_org, target_org) in enumerate(data_loader):
        batch_size = data_org.shape[0] // 2
        data_org, target_org = data_org.to(device), target_org.float()
        optimizer_ag.zero_grad()

        for t in range(args.classes_per_set): # for each task (class)
            data, target = data_org[t*args.samples_per_class*2: (t+1)*args.samples_per_class*2], \
                           target_org[t * args.samples_per_class*2: (t + 1) * args.samples_per_class*2]
            data = data.view(2, args.samples_per_class, 1, 84, 84)
            target = target.view(2, args.samples_per_class, -1)
            support_x, target_x = data[0, :, :, :, :].reshape(args.samples_per_class, 1, 84, 84), \
                                  data[1, :, :, :, :].reshape(args.samples_per_class, 1, 84, 84)
            support_y, target_y = target[0, :, :].reshape(-1, 5), \
                                  target[1, :, :].reshape(-1, 5)
            '''perform updates for this task'''
            fast_parameters = list(agent.parameters()) #the first gradient calcuated is based on original weight
            for weight in agent.parameters():
                weight.fast = None
            agent.zero_grad()
            with torch.no_grad():
                _, _, _, _, iou = localize(model, agent, target_x, target_y)
                corrects[0] += (iou >= 0.5).sum()
            for task_step in range(args.task_update_num):
                set_loss, q, loss_ent, rewards_deltas, iou = \
                    localize(model, agent, support_x, support_y, task_step)
                grad = torch.autograd.grad(set_loss, fast_parameters,
                            create_graph=True)  # build full graph support gradient of gradient
                if args.approx == 1:
                    grad = [g.detach() for g in
                            grad]  # do not calculate gradient of gradient if using first order approximation
                fast_parameters = []
                for k, weight in enumerate(agent.parameters()):
                    #if k == 0: print(weight.fast is None)
                    # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py
                    if weight.fast is None:
                        weight.fast = weight - args.train_lr * grad[k]  # create weight.fast
                    else:
                        weight.fast = weight.fast - args.train_lr * grad[
                            k]  # create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast
                    fast_parameters.append(
                        weight.fast)  # gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts

            # weight.fast is not None
            set_loss, q, loss_ent, rewards_deltas, iou = localize(model, agent, target_x, target_y) # compute loss using updated model
            corrects[1] += (iou >= 0.5).sum()
            ''''''

            loss_all.append(set_loss)
            pol_losses.append(q.sum().item())
            ent_losses.append(loss_ent.sum().item())
            rewards.append(torch.mean(rewards_deltas).item())

            task_count += 1

            if task_count == args.n_task:
                loss_q = torch.stack(loss_all).sum(0)
                loss_q.backward()
                optimizer_ag.step()
                task_count = 0
                loss_all = []

        if batch_idx % args.log_interval == 0:
            if batch_idx == 0:
                print('\n')
            print('{} Epoch: {} [{}/{} ({:.0f}%)]\tReward: {:.4f}'.format(
                phase, epoch, batch_idx * batch_size,
                len(data_loader.dataset),
                100. * batch_idx / len_dataloader, np.mean(rewards)))
        #break

    total_pol_loss = np.mean(pol_losses)
    iterations = args.iterations if phase == 'Train' else args.iterations_test
    total_img = iterations * args.samples_per_class * args.classes_per_set
    acc_bf = 100. * corrects[0] / total_img
    acc_af = 100. * corrects[1] / total_img
    log_message = {"{} Pol Loss".format(phase): total_pol_loss,
                   '{} Rewards'.format(phase): np.mean(rewards),
                   '{} Loc BF Acc'.format(phase): acc_bf,
                   '{} Loc AF Acc'.format(phase): acc_af,
                   '{} Ent Loss'.format(phase): np.mean(ent_losses)
                   }

    if phase == 'Train':
        current_lr = scheduler_ag.get_last_lr()[0]
        log_message.update({'learning rate': current_lr})
    if upload:
        wandb.log(log_message, step=epoch)

    return total_pol_loss, acc_af


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

    best_acc = 0.
    save_model = False
    no_improve_epoch = 0
    for epoch in range(args.epochs):
        train_log = feed_data(net, agent, data_loader=train_loader, is_train=True)
        scheduler_ag.step()

        #finetune_log = feed_data(net, agent, data_loader=test_loader, is_train=True, upload=False)
        #with torch.no_grad():
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




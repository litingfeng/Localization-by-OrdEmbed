"""
Train agent using ordinal embedding
Created on 4/6/2021 4:08 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import os, wandb
import torch, copy
import random
from util.utils import box_iou
import math
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from transform import MyBoxScaleTransform
from util.augmentations import Compose
from util.data_aug import Resize
from util.actions_tree_dqn_joint import Actor
from opts import parser
from util.replay import Transition, ReplayMemory
from datasets.clutter_mnist_scale_rl import MNIST_CoLoc
from models.mnist_scale_dqn_model import Net, Agent
#from models.mnist_baseline_ae_model import EmbeddingNet, Agent

DEBUG = True
def select_action(state, policy_net, epoch, n_iter, h_t_prev=None):
    sample = np.random.random(state.shape[0])
    eps_threshold = epsilons[min(epoch, args.eps_decay_steps-1)]
    # eps_threshold = args.eps_end + (args.eps_start - args.eps_end) * \
    #                 math.exp(-1. * n_iter / args.eps_decay_steps)
    actions = torch.randint(0, args.num_act, (state.shape[0],)).to(device)
    pol_inds = sample > eps_threshold
    if pol_inds.sum() != 0:
        with torch.no_grad():
            if h_t_prev is not None:
                h_t, logits = policy_net(state[pol_inds], h_t_prev)
            else:
                logits = policy_net(state[pol_inds])
            actions[pol_inds] = logits.max(1)[1]
    if h_t_prev is not None:
        return actions, h_t
    return actions, eps_threshold

def init_dataloader():
    kwargs = {'num_workers': 8, 'pin_memory': True}

    train_transform = Compose([Resize(84)])
    test_transform = Compose([Resize(84)])
    trainset = MNIST_CoLoc(root='.', train=True, digit=args.digit,
                           datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                           clutter=1, transform=train_transform)
    testset = MNIST_CoLoc(root='.', train=False, digit=args.digit,
                          datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                          clutter=1, transform=test_transform)

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
    # if load pretrained encoder
    #assert (args.rnn == 0)
    if args.pretrained != '':
        ckpt = torch.load(os.path.join('/research/cbim/vast/tl601/results/'
                          'selfpaced/ordinal/',
                          args.pretrained))
        net.load_state_dict(ckpt['state_dict'])
        print('loaded from {}\nckpt epoch {} acc {:.2f}%'.format(args.pretrained,
                                                    ckpt['epoch'], ckpt['acc']))
    if args.freeze == 1:
        print('freezing embedding net')
        for name, value in net.named_parameters():
            value.requires_grad = False

    agent = Agent(rnn=args.rnn, dim=512, num_class=args.num_act,
                  hist_len=args.hist_len,
                  hidden_size=args.hidden_size).to(device)
    agent_target = Agent(rnn=args.rnn, dim=512, num_class=args.num_act,
                         hist_len=args.hist_len,
                         hidden_size=args.hidden_size).to(device)
    # agent = Agent(dim=args.pooling_size, hidden=args.hidden_size,
    #               history=args.hist, hist_len=args.hist_len,
    #               num_class=args.num_act).to(device)
    # agent_target = Agent(dim=args.pooling_size, hidden=args.hidden_size,
    #               history=args.hist, hist_len=args.hist_len,
    #               num_class=args.num_act).to(device)
    agent_target.load_state_dict(agent.state_dict())
    agent_target.eval()

    return net, agent, agent_target

# def init_model():
#     embeddingnet = EmbeddingNet(pooling_size=args.pooling_size,
#                    pooling_mode=args.pooling_mode).to(device)
#     embeddingnet_dict = embeddingnet.state_dict()
#     # Fiter out unneccessary keys
#     mnist_pretrained_dict = torch.load('/research/cbim/vast/tl601/results/'
#                                        'mnist/autoencoder_4_model.pth')
#
#     filtered_dict = {}
#     for k, v in mnist_pretrained_dict.items():
#         if k in embeddingnet_dict:
#             filtered_dict[k] = v
#     embeddingnet_dict.update(filtered_dict)
#     embeddingnet.load_state_dict(embeddingnet_dict)
#     print('loaded pretained MNIST encoder')
#
#     if args.freeze:
#         print('freezing embedding net')
#         for name, value in embeddingnet.named_parameters():
#             value.requires_grad = False
#
#     agent = Agent(dim=args.pooling_size, hidden=args.hidden_size,
#                   history=args.hist, hist_len=args.hist_len,
#                   num_class=args.num_act).to(device)
#     agent_target = Agent(dim=args.pooling_size, hidden=args.hidden_size,
#                   history=args.hist, hist_len=args.hist_len,
#                   num_class=args.num_act).to(device)
#     agent_target.load_state_dict(agent.state_dict())
#     agent_target.eval()
#
#     return embeddingnet, agent, agent_target

def train(policy_net, target_net, embedding_net, data_loader):
    policy_net.train()
    #embedding_net.train()
    global n_iter

    total_pol_loss, total_rewards,ord_losses = [], [], [0]
    correct_ord_inst, correct_ord_proto, correct_reward = 0, 0, 0
    num_located = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        batch_size = data.shape[0]
        data, target = data.to(device), target.float()
        org_data = data.clone()
        optimizer_ag.zero_grad()

        '''train agent'''
        # initialize
        pred_boxes = torch.tensor([[0, 0,
                                    83, 83]]).float().repeat(batch_size, 1)
        if args.hist == 1:
            action_history = torch.zeros(batch_size,
                                         args.hist_len, args.num_act).to(device)
        if args.rnn == 1:
            h_t = torch.zeros(
                batch_size,
                args.hidden_size,
                dtype=torch.float,
                device=device,
                requires_grad=True,
            )

        rois = torch.cat((torch.arange(0, batch_size).float().view(-1, 1),
                          pred_boxes), dim=1).to(device)
        actor = Actor(data, target, min_box_side=28)
        pol_losses, rewards_stps, boxes = 0, [], []

        with torch.no_grad():
            state = embedding_net.get_embedding(data, rois)  # (bs, dim)
        state = torch.cat((state, action_history.view(batch_size, -1)), 1) \
            if args.hist == 1 else state
        # compute target region embedding
        rois_t0 = torch.cat((torch.arange(0, batch_size).float().view(-1, 1),
                             target), dim=1).to(device)
        org_embed_t = embedding_net.get_embedding(data, rois_t0)
        embed_t = org_embed_t.clone().detach()
        prev_dist = F.pairwise_distance(embed_t, state[:,:512]).cpu()
        all_ious, all_feat, all_ids = [], [], []
        ids = torch.arange(0, batch_size)
        for t in range(args.seq_len):
            if args.rnn == 1:
                actions, h_t = select_action(state, policy_net, epoch, n_iter, h_t_prev=h_t)
            else:
                actions, eps = select_action(state, policy_net, epoch, n_iter)
            #print(t, ' actions ', actions)
            # update action history
            if args.hist == 1: #TODO modify to queue
                action_history[:, t%args.hist_len, :] = torch.nn.functional.one_hot(actions,
                                                        num_classes=args.num_act)
            rewards, new_boxes = actor.takeaction(actions)
            num_located += (rewards == 3).sum()
            rois = torch.cat((torch.arange(0, state.shape[0]).float().view(-1, 1),
                              new_boxes), dim=1).float().to(device)
            with torch.no_grad():
                next_state = embedding_net.get_embedding(data, rois)
            done = torch.zeros_like(actions).cpu()
            done[actions == 5] = 1
            done_inds = (done == 0)
            # print('done ', done)
            # print('done_inds ', done_inds)
            boxes.append(new_boxes)
            all_ids.append(ids)
            all_ious.append(actor.ious.clone())

            ''' compute embedding reward'''
            if args.reward_mode == 'embed':
                dist = F.pairwise_distance(embed_t, next_state[:,:512]).cpu()
                if t == 5:
                    print('iou rewards ', rewards[:10])
                #print('dist ', dist.shape)
                iou_reward = rewards.clone()
                rewards = - (dist - prev_dist)
                if t == 5:
                    print('embed rewards ', rewards[:10])
                sign_rewards = torch.sign(rewards.clone())
                sign_rewards[sign_rewards == 0] = -1.
                iou = torch.diagonal(box_iou(new_boxes.float(),
                                             actor.targets), 0)
                sign_rewards[torch.logical_and(done==1, iou>0.5)] = 3
                sign_rewards[torch.logical_and(done==1, iou<=0.5)] = -3
                correct_reward += (iou_reward == sign_rewards).sum()
                if args.sign == 1:
                    rewards = sign_rewards
                if t == 5:
                    print('sign rewards ', rewards[:10])
                prev_dist = dist

            next_state = torch.cat((next_state,
                                    action_history.view(action_history.shape[0], -1)), 1) \
                         if args.hist == 1 else next_state

            # state (num_live, dim), actions (num_live, ),
            # next_state (num_live, dim), rewards (num_live, ), done (num_live,)
            memory.push_batch(state, actions, next_state, rewards, done)

            state = next_state[done_inds]
            data = data[done_inds]
            actor.ious = actor.ious[done_inds]
            actor.targets = actor.targets[done_inds]
            actor.agent_windows = torch.from_numpy(actor.agent_windows)[done_inds].numpy()
            embed_t = embed_t[done_inds]
            prev_dist = prev_dist[done_inds]
            ids = ids[done_inds]
            if args.hist == 1:
                action_history = action_history[done_inds]

            # Perform one step of the optimization
            loss = optimize_model(policy_net, target_net)
            pol_losses += loss.item()
            n_iter += 1

            rewards_stps += [rewards]

            # Update the target network, copying all weights and biases in DQN
            if epoch % args.update_target_every == 0:
            #if n_iter % args.update_target_every == 0:
                target_net.load_state_dict(policy_net.state_dict())

            total_pol_loss.append(pol_losses)

            if state.shape[0] == 0:
                break
        #break
        #exit()
        total_rewards.append(torch.cat(rewards_stps))
        '''train embedding net'''
        # optimizer.zero_grad()
        # sample_dict = {i: [] for i in range(batch_size)}
        # for id, iou, box in zip(all_ids, all_ious, boxes):
        #     for i in range(id.shape[0]):
        #         sample_dict[id[i].item()].append((box[i], iou[i]))
        #
        # # sample two boxes from each sequence
        # smp_boxes, smp_ious, smp_ids = [], [], []
        # for i in range(batch_size):
        #     if len(sample_dict[i]) < 2:
        #         continue
        #     ind = np.random.choice(len(sample_dict[i]), size=2, replace=False)
        #     smp_boxes += [sample_dict[i][ind[0]][0], sample_dict[i][ind[1]][0]]
        #     smp_ious += [sample_dict[i][ind[0]][1], sample_dict[i][ind[1]][1]]
        #     smp_ids.append(i)
        # smp_boxes = torch.stack(smp_boxes).view(-1, 2, 4).to(device)
        # smp_ious = torch.stack(smp_ious).view(-1, 2)
        # smp_ids = np.array(smp_ids)
        # data, target = org_data[smp_ids], target[smp_ids].to(device)
        # embed_t = org_embed_t[smp_ids]
        #
        # bs = len(smp_ids)
        # inds = torch.arange(0, bs).float().view(-1, 1).repeat(1,
        #                     2).to(device)
        # rois = torch.cat((inds.view(-1, 1),
        #                   smp_boxes.view(-1, 4)), dim=1)  # (num*2, 5)
        # feature = embedding_net.get_embedding(data, rois.detach()).view(bs, 2, -1)  # (bs, 2, dim)
        #
        # # compute loss order
        # inds = torch.max(smp_ious, 1).indices
        # p = feature[torch.arange(0, bs), inds, :]
        # n = feature[torch.arange(0, bs), 1 - inds, :]
        # proto = torch.mean(embed_t, dim=0).view(1, -1).repeat(bs, 1)
        # # loss = triplet_loss(proto, p, n)
        # loss_ord = 0 #triplet_loss(embed_t, p, n)
        #
        # # compute accuracy
        # d_a0a1 = F.pairwise_distance(embed_t, feature[:, 0, :])
        # d_a0a2 = F.pairwise_distance(embed_t, feature[:, 1, :])
        # sub_pix = smp_ious[:, 1] - smp_ious[:, 0]
        # sub_d = d_a0a1 - d_a0a2
        # mask = (torch.sign(sub_d) == torch.sign(sub_pix.to(device)))
        # correct_ord_inst += mask.sum().item()
        #
        # d_a0a1 = F.pairwise_distance(proto, feature[:, 0, :])
        # d_a0a2 = F.pairwise_distance(proto, feature[:, 1, :])
        # sub_d = d_a0a1 - d_a0a2
        # mask = (torch.sign(sub_d) == torch.sign(sub_pix.to(device)))
        # correct_ord_proto += mask.sum().item()
        #
        # # update model
        # ord_losses.append(loss_ord.item())
        # # loss_ord.backward()
        # # optimizer.step()

            #print(t, ' total rewards ', total_rewards)
    acc = 100. * num_located / len(data_loader.dataset)
    acc_inst = 100. * correct_ord_inst / len(data_loader.dataset)
    acc_proto = 100. * correct_ord_proto / len(data_loader.dataset)
    acc_r = 100. * correct_reward / (len(data_loader.dataset) * args.seq_len)

    log_message = {"Train Rewards": torch.mean(torch.cat(total_rewards)).item(),
                   "Train Loc Acc": acc,
                   "Train Ord Loss": np.mean(ord_losses),
                   'Train Ord Acc Inst': acc_inst,
                   'Train Ord Acc Proto': acc_proto,
                   'Train Reward Acc': acc_r,
                   'Train epsilon': eps
                   }

    current_lr = scheduler_ag.get_last_lr()[0]
    log_message.update({'learning rate': current_lr,
                        "Train Pol Loss": np.mean(total_pol_loss)})
    wandb.log(log_message, step=epoch)

    return total_pol_loss, acc

def evaluate(policy_net, embedding_net, data_loader):
    policy_net.eval()
    embedding_net.eval()

    correct_loc, correct_ord_inst, correct_ord_proto = 0, 0, 0
    total_rewards = []
    correct_reward = 0
    ord_losses = [0]
    iouis, ioujs = [], []
    for batch_idx, (data, target) in enumerate(data_loader):
        batch_size = data.shape[0]
        data, target = data.to(device), target.float()
        org_data = data.clone()

        pred_boxes = torch.tensor([[0, 0,
                                    83, 83]]).float().repeat(batch_size, 1).to(device)
        rois = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                          pred_boxes), dim=1)
        if args.rnn == 1:
            h_t = torch.zeros(
                batch_size,
                args.hidden_size,
                dtype=torch.float,
                device=device,
                requires_grad=True,
            )
        actor = Actor(data, target, min_box_side=28)
        if args.hist == 1:
            action_history = torch.zeros(batch_size,
                                         args.hist_len, args.num_act).to(device)

        state = embedding_net.get_embedding(data, rois)
        state = torch.cat((state, action_history.view(batch_size, -1)), 1) \
            if args.hist == 1 else state
        action_seq = torch.IntTensor(batch_size, args.seq_len).to(device)
        rewards_all = torch.zeros((batch_size, args.seq_len)).to(device)
        boxes, all_ids, all_ious = [], [], []

        # compute target region embedding
        rois_t0 = torch.cat((torch.arange(0, batch_size).float().view(-1, 1),
                             target), dim=1).to(device)
        org_embed_t = embedding_net.get_embedding(data, rois_t0)
        embed_t = org_embed_t.clone()
        prev_dist = F.pairwise_distance(embed_t, state[:,:512]).cpu()
        ids = torch.arange(0, batch_size)
        for t in range(args.seq_len): # TODO stop action
            if args.rnn == 1:
                h_t, actions = policy_net(state, h_t_prev=h_t).max(1)[1]
            else:
                actions = policy_net(state).max(1)[1]
            rewards, new_boxes = actor.takeaction(actions)
            rois = torch.cat((torch.arange(0, state.shape[0]).float().view(-1, 1),
                              new_boxes), dim=1).float().to(device)
            with torch.no_grad():
                state = embedding_net.get_embedding(data, rois)

            ''' compute embedding reward'''
            if args.reward_mode == 'embed':
                done = torch.zeros_like(actions).cpu()
                done[actions == 5] = 1
                dist = F.pairwise_distance(embed_t, state[:,:512]).cpu()
                iou_reward = rewards.clone()
                rewards = - (dist - prev_dist)
                # compute sign rewards
                sign_rewards = torch.sign(rewards.clone())
                sign_rewards[sign_rewards == 0] = -1.
                iou = torch.diagonal(box_iou(new_boxes.float(),
                                             actor.targets), 0)
                sign_rewards[torch.logical_and(done == 1, iou > 0.5)] = 3
                sign_rewards[torch.logical_and(done == 1, iou <= 0.5)] = -3
                correct_reward += (iou_reward == sign_rewards).sum()
                if args.sign == 1:
                    rewards = sign_rewards
                prev_dist = dist

            state = torch.cat((state,
                               action_history.view(batch_size, -1)), 1) \
                if args.hist == 1 else state
            boxes.append(new_boxes)
            all_ids.append(ids)
            all_ious.append(actor.ious.clone())
            action_seq[:, t] = actions.view(1, -1)
            rewards_all[:, t] = rewards

        '''evaluate embedding net'''
        # sample_dict = {i: [] for i in range(batch_size)}
        # for id, iou, box in zip(all_ids, all_ious, boxes):
        #     for i in range(id.shape[0]):
        #         sample_dict[id[i].item()].append((box[i], iou[i]))
        #
        # # sample two boxes from each sequence
        # smp_boxes, smp_ious, smp_ids = [], [], []
        # for i in range(batch_size):
        #     if len(sample_dict[i]) < 2:
        #         continue
        #     ind = np.random.choice(len(sample_dict[i]), size=2, replace=False)
        #     smp_boxes += [sample_dict[i][ind[0]][0], sample_dict[i][ind[1]][0]]
        #     smp_ious += [sample_dict[i][ind[0]][1], sample_dict[i][ind[1]][1]]
        #     smp_ids.append(i)
        # smp_boxes = torch.stack(smp_boxes).view(-1, 2, 4).to(device)
        # smp_ious = torch.stack(smp_ious).view(-1, 2)
        # smp_ids = np.array(smp_ids)
        # data, target = org_data[smp_ids], target[smp_ids]
        # embed_t = org_embed_t[smp_ids]
        #
        # bs = len(smp_ids)
        # #print('bs ', bs)
        # inds = torch.arange(0, bs).float().view(-1, 1).repeat(1,
        #                                                       2).to(device)
        # rois = torch.cat((inds.view(-1, 1),
        #                   smp_boxes.view(-1, 4)), dim=1)  # (num*2, 5)
        # feature = embedding_net.get_embedding(data, rois.detach()).view(bs, 2, -1)  # (bs, 2, 1024)
        #
        # # compute loss order
        # inds = torch.max(smp_ious, 1).indices
        # p = feature[torch.arange(0, bs), inds, :]
        # n = feature[torch.arange(0, bs), 1 - inds, :]
        # proto = torch.mean(embed_t, dim=0).view(1, -1).repeat(bs, 1)
        # # loss = triplet_loss(proto, p, n)
        # #loss_ord = triplet_loss(embed_t, p, n)
        # #ord_losses.append(loss_ord.item())
        # iouis.append(smp_ious[:, 0])
        # ioujs.append(smp_ious[:, 1])
        #
        # # compute accuracy
        # d_a0a1 = F.pairwise_distance(embed_t, feature[:, 0, :])
        # d_a0a2 = F.pairwise_distance(embed_t, feature[:, 1, :])
        # sub_pix = smp_ious[:, 1] - smp_ious[:, 0]
        # sub_d = d_a0a1 - d_a0a2
        # mask = (torch.sign(sub_d) == torch.sign(sub_pix.to(device)))
        # correct_ord_inst += mask.sum().item()
        #
        # d_a0a1 = F.pairwise_distance(proto, feature[:, 0, :])
        # d_a0a2 = F.pairwise_distance(proto, feature[:, 1, :])
        # sub_d = d_a0a1 - d_a0a2
        # mask = (torch.sign(sub_d) == torch.sign(sub_pix.to(device)))
        # correct_ord_proto += mask.sum().item()

        # calculate accuracy
        final_iou = torch.diagonal(box_iou(new_boxes.float(),
                                           target), 0)
        correct_loc += (final_iou >= 0.5).sum()
        total_rewards.append(rewards_all.cpu())

        if DEBUG:
            if batch_idx == 0:
                print('\n')
            print('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * batch_size, len(data_loader.dataset),
                              100. * batch_idx / len(data_loader)))
            print('action_seq ', action_seq[:5])
            print('rewards_all ', rewards_all[:5])
            print('pred_boxes ', new_boxes[:5])
            print('target_boxes ', target[:5].int())

    acc = 100. * correct_loc / len(data_loader.dataset)
    acc_inst = 100. * correct_ord_inst / len(data_loader.dataset)
    acc_proto = 100. * correct_ord_proto / len(data_loader.dataset)
    acc_r = 100. * correct_reward / (len(data_loader.dataset) * args.seq_len)

    # iouis = torch.cat(iouis).cpu().numpy()
    # ioujs = torch.cat(ioujs).cpu().numpy()
    # f, ax = plt.subplots(2, 1)
    # ax[0].hist(iouis, label='iouis')
    # ax[1].hist(ioujs, label='ioujs')
    # ax[0].legend()
    # ax[1].legend()

    log_message = {"Test Ord Loss": np.mean(ord_losses),
                  "Test Loc Acc": acc,
                   'Test Rewards': torch.mean(torch.cat(total_rewards)).item(),
                   "Test Ord Acc Inst": acc_inst,
                   "Test Ord Acc Proto": acc_proto,
                   '{} Reward Acc'.format("Test"): acc_r,
                   'Test IoU Distribution': wandb.Image(plt)}
    wandb.log(log_message, step=epoch)

    return acc

def optimize_model(policy_net, target_net):
    if len(memory) < args.batch_size:
        return 0
    transitions = memory.sample(args.batch_size)
    batch = Transition(*zip(*transitions))

    state_batch = torch.stack(batch.state)
    next_state_batch = torch.stack(batch.next_state)
    action_batch = torch.stack(batch.action).unsqueeze(1)
    reward_batch = torch.stack(batch.reward).to(device).float()
    done_batch = torch.stack(batch.done).to(device)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Calculate q values and targets
    q_values_next = target_net(next_state_batch).max(1)[0].detach()
    expected_state_action_values = reward_batch + (1-done_batch) * args.gamma * q_values_next

    # Compute Huber loss
    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    #loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer_ag.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer_ag.step()

    return loss

if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ['WANDB_NAME'] = args.savename
    wandb.init(project="selfpaced")
    saveroot = os.path.join('/research/cbim/vast/tl601/results/selfpaced/ordinal',
                            args.savename)
    os.makedirs(saveroot, exist_ok=True)

    train_loader, test_loader = init_dataloader()

    embedding_net, policy_net, target_net = init_model()
    wandb.watch(policy_net, log_freq=10)
    wandb.watch(embedding_net, log_freq=10)
    wandb.config.update(args)

    triplet_loss = nn.TripletMarginLoss(margin=args.margin, p=2)
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(embedding_net.parameters(),
                                    lr=args.lr, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(embedding_net.parameters(),
                                     lr=args.lr)
    if args.optimizer_ag == 'SGD':
        optimizer_ag = torch.optim.SGD(policy_net.parameters(), lr=args.lr_ag, momentum=0.9)
    elif args.optimizer_ag == 'Adam':
        optimizer_ag = torch.optim.Adam(policy_net.parameters(), lr=args.lr_ag)
    scheduler_ag = torch.optim.lr_scheduler.MultiStepLR(optimizer_ag,
                                                   milestones=[args.step_ag], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=args.steps, gamma=0.1)
    # scheduler_ag = torch.optim.lr_scheduler.StepLR(optimizer_ag,
    #                                                step_size=args.step_ag, gamma=0.1)

    if args.evaluate == 1:
        policy_path = agent_path = os.path.join(saveroot, 'best.pth.tar')
        # load agent
        print('loading net ckpt from ', agent_path)
        ckpt = torch.load(policy_path)
        state_dict = ckpt['state_dict']
        epoch = ckpt['epoch']
        best_acc = ckpt['acc']
        policy_net.load_state_dict(state_dict)
        print(("=> loaded agent checkpoint epoch {} {}".format(epoch, best_acc)))

        with torch.no_grad():
            test_acc = evaluate(policy_net, embedding_net, test_loader)
        print(args.digit, ' test_acc ', test_acc)
        exit()


    memory = ReplayMemory(args.buffer_size)
    # The epsilon decay schedule
    epsilons = np.linspace(args.eps_start, args.eps_end, args.eps_decay_steps)

    best_acc = 0.
    save_model = False
    no_improve_epoch = 0
    n_iter = 0
    for epoch in range(args.epochs):
        train_log = train(policy_net, target_net, embedding_net,
                          data_loader=train_loader)
        scheduler_ag.step()
        scheduler.step()

        with torch.no_grad():
            test_acc = evaluate(policy_net, embedding_net, test_loader)

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
                    'state_dict': policy_net.state_dict(),
                    'acc': test_acc
                }, os.path.join(saveroot, 'best_agent.pth.tar'))
                torch.save({
                    'epoch': epoch,
                    'state_dict': embedding_net.state_dict(),
                    'acc': test_acc
                }, os.path.join(saveroot, 'best_net.pth.tar'))

            # break training
            if no_improve_epoch > args.patience:
                print('stop training...')
                break

    torch.save({
        'epoch': epoch,
        'state_dict': policy_net.state_dict(),
        'acc': test_acc
    }, os.path.join(saveroot, 'last_agent.pth.tar'))
    torch.save({
        'epoch': epoch,
        'state_dict': embedding_net.state_dict(),
        'acc': test_acc
    }, os.path.join(saveroot, 'last_net.pth.tar'))








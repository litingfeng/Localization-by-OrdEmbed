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
from util.actions_tree_dqn_batch import Actor
from opts import parser
from util.replay import Transition, ReplayMemory
from datasets.clutter_mnist_scale_rl import MNIST_CoLoc
from models.mnist_scale_dqn_model import Net, Agent

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
    assert (args.freeze == 1)
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
    agent_target.load_state_dict(agent.state_dict())
    agent_target.eval()

    return net, agent, agent_target

def train(policy_net, target_net, embedding_net, data_loader):
    policy_net.train()
    global n_iter

    total_pol_loss, total_rewards = [], []
    num_located = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        batch_size = data.shape[0]
        data, target = data.to(device), target.float()
        optimizer_ag.zero_grad()

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
            state = embedding_net(data, rois)  # (bs, dim)
        state = torch.cat((state, action_history.view(batch_size, -1)), 1) \
            if args.hist == 1 else state
        # compute target region embedding
        rois_t0 = torch.cat((torch.arange(0, batch_size).float().view(-1, 1),
                             target), dim=1).to(device)
        with torch.no_grad():
            embed_t = embedding_net(data, rois_t0)
        prev_dist = F.pairwise_distance(embed_t, state[:,:512]).cpu()
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
                next_state = embedding_net(data, rois)
            done = torch.zeros_like(actions).cpu()
            done[actions == 5] = 1
            done_inds = (done == 0)
            # print('done ', done)
            # print('done_inds ', done_inds)

            ''' compute embedding reward'''
            if args.reward_mode == 'embed':
                dist = F.pairwise_distance(embed_t, next_state[:,:512]).cpu()
                if t == 5:
                    print('iou rewards ', rewards[:10])
                #print('dist ', dist.shape)
                rewards = - (dist - prev_dist)
                if t == 5:
                    print('embed rewards ', rewards[:10])
                if args.sign == 1:
                    rewards = torch.sign(rewards)
                    iou = torch.diagonal(box_iou(new_boxes.float(),
                                                 actor.targets), 0)
                    rewards[torch.logical_and(done==1, iou>0.5)] = 3
                    rewards[torch.logical_and(done==1, iou<=0.5)] = -3
                    if t == 5:
                        print('sign rewards ', rewards[:10])
                # print('iou ', iou)
                # print('rewards ', rewards)
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
            if args.hist == 1:
                action_history = action_history[done_inds]

            # Perform one step of the optimization
            loss = optimize_model(policy_net, target_net)
            pol_losses += loss.item()
            n_iter += 1

            rewards_stps += [rewards]
            boxes.append(new_boxes)

            # Update the target network, copying all weights and biases in DQN
            if epoch % args.update_target_every == 0:
            #if n_iter % args.update_target_every == 0:
                target_net.load_state_dict(policy_net.state_dict())

            total_pol_loss.append(pol_losses)
            total_rewards.append(torch.mean(torch.cat(rewards_stps)))
            if state.shape[0] == 0:
                break
        #break
        #exit()
            #print(t, ' total rewards ', total_rewards)
    acc = 100. * num_located / len(data_loader.dataset)
    log_message = {"Train Rewards": np.mean(total_rewards),
                   "Train Loc Acc": acc,
                   'Train epsilon': eps}

    current_lr = scheduler_ag.get_last_lr()[0]
    log_message.update({'learning rate': current_lr,
                        "Train Pol Loss": np.mean(total_pol_loss)})
    wandb.log(log_message, step=epoch)

    return total_pol_loss, acc

def evaluate(policy_net, embedding_net, data_loader):
    policy_net.eval()

    correct_loc = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        batch_size = data.shape[0]
        data, target = data.to(device), target.float()

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
        actor = Actor(data, target.cpu(), min_box_side=28)
        if args.hist == 1:
            action_history = torch.zeros(batch_size,
                                         args.hist_len, args.num_act).to(device)
        state = embedding_net(data, rois)
        state = torch.cat((state, action_history.view(batch_size, -1)), 1) \
            if args.hist == 1 else state
        action_seq = torch.IntTensor(batch_size, args.seq_len).to(device)
        rewards_all = torch.zeros((batch_size, args.seq_len)).to(device)
        boxes = []
        # compute target region embedding
        rois_t0 = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                             target.to(device)), dim=1)
        embed_t = embedding_net(data, rois_t0)
        prev_dist = F.pairwise_distance(embed_t, state[:,:512]).cpu()
        for t in range(args.seq_len): # TODO stop action
            if args.rnn == 1:
                h_t, actions = policy_net(state, h_t_prev=h_t).max(1)[1]
            else:
                actions = policy_net(state).max(1)[1]
            rewards, new_boxes = actor.takeaction(actions)
            rois = torch.cat((torch.arange(0, state.shape[0]).float().view(-1, 1),
                              new_boxes), dim=1).float().to(device)
            state = embedding_net(data, rois)

            ''' compute embedding reward'''
            if args.reward_mode == 'embed':
                done = torch.zeros_like(actions).cpu()
                done[actions == 5] = 1
                dist = F.pairwise_distance(embed_t, state[:,:512]).cpu()
                rewards = - (dist - prev_dist)
                if args.sign == 1:
                    rewards = torch.sign(rewards)
                    iou = torch.diagonal(box_iou(new_boxes.float(),
                                                 actor.targets), 0)
                    rewards[torch.logical_and(done == 1, iou > 0.5)] = 3
                    rewards[torch.logical_and(done == 1, iou <= 0.5)] = -3
                prev_dist = dist

            state = torch.cat((state,
                               action_history.view(batch_size, -1)), 1) \
                if args.hist == 1 else state
            boxes.append(new_boxes)
            action_seq[:, t] = actions.view(1, -1)
            rewards_all[:, t] = rewards

        # calculate accuracy
        final_iou = torch.diagonal(box_iou(new_boxes.float(),
                                           target), 0)
        correct_loc += (final_iou >= 0.5).sum()

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
    log_message = {"Test Loc Acc": acc}
    wandb.log(log_message, step=epoch)

    return acc

def optimize_model(policy_net, target_net):
    if len(memory) < args.batch_size:
        return
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
    wandb.config.update(args)

    if args.optimizer_ag == 'SGD':
        optimizer_ag = torch.optim.SGD(policy_net.parameters(), lr=args.lr_ag, momentum=0.9)
    elif args.optimizer_ag == 'Adam':
        optimizer_ag = torch.optim.Adam(policy_net.parameters(), lr=args.lr_ag)
    elif args.optimizer_ag == 'RMS':
        optimizer_ag = torch.optim.RMSprop(policy_net.parameters(), lr=args.lr_ag)
    scheduler_ag = torch.optim.lr_scheduler.MultiStepLR(optimizer_ag,
                                                   milestones=[args.step_ag], gamma=0.1)
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

        kwargs = {'num_workers': 8, 'pin_memory': True}
        test_transform = Compose([Resize(84)])
        test_accs = []
        with torch.no_grad():
            for d in range(0, 10):
                if d == 4:
                    continue
                testset = MNIST_CoLoc(root='.', train=False, digit=d,
                                      datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                                      clutter=1, transform=test_transform
                                      )
                test_loader = torch.utils.data.DataLoader(
                    testset, batch_size=args.batch_size, shuffle=False, **kwargs)
                test_acc = evaluate(policy_net, embedding_net, test_loader)
                test_accs.append(test_acc)
                print(d, ' test_acc ', test_acc)
            print('mean acc ', np.mean(test_accs))
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
                }, os.path.join(saveroot, 'best.pth.tar'))

            # break training
            if no_improve_epoch > args.patience:
                print('stop training...')
                break

    torch.save({
        'epoch': epoch,
        'state_dict': policy_net.state_dict(),
        'acc': test_acc
    }, os.path.join(saveroot, 'last.pth.tar'))








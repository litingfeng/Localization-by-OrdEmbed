"""
baseline DQN model for clutter mnist localization
Created on 3/31/2021 2:42 PM

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
from models.mnist_baseline_ae_model import EmbeddingNet, Agent

DEBUG = True

def select_action(state, policy_net, epoch, n_iter):
    sample = np.random.random(state.shape[0])
    eps_threshold = epsilons[min(epoch, args.eps_decay_steps-1)]
    # eps_threshold = args.eps_end + (args.eps_start - args.eps_end) * \
    #                 math.exp(-1. * n_iter / float(args.eps_decay_steps))
    actions = torch.randint(0, args.num_act, (state.shape[0],)).to(device)
    pol_inds = sample > eps_threshold
    if pol_inds.sum() != 0:
        with torch.no_grad():
            logits = policy_net(state[pol_inds])
            actions[pol_inds] = logits.max(1)[1]
    return actions, eps_threshold

def init_dataloader():
    kwargs = {'num_workers': 8, 'pin_memory': True}

    train_transform = Compose([Resize(84)])
    test_transform = Compose([Resize(84)])
    trainset = MNIST_CoLoc(root='.', train=True, digit=args.digit, sample_size=args.sample_size,
                           datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                           clutter=1, transform=train_transform
                           )
    testset = MNIST_CoLoc(root='.', train=False, digit=args.digit,
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

def init_model():
    embeddingnet = EmbeddingNet(pooling_size=args.pooling_size,
                   pooling_mode=args.pooling_mode).to(device)
    embeddingnet_dict = embeddingnet.state_dict()
    # Fiter out unneccessary keys
    model_path = '/research/cbim/vast/tl601/results/' \
                                       'mnist/{}.pth'.format(args.pretrained)
    mnist_pretrained_dict = torch.load(model_path)

    filtered_dict = {}
    for k, v in mnist_pretrained_dict.items():
        if k in embeddingnet_dict:
            filtered_dict[k] = v
    embeddingnet_dict.update(filtered_dict)
    embeddingnet.load_state_dict(embeddingnet_dict)
    print('loaded pretained MNIST encoder ', model_path)

    assert (args.freeze == 1)
    if args.freeze:
        print('freezing embedding net')
        for name, value in embeddingnet.named_parameters():
            value.requires_grad = False

    agent = Agent(dim=args.pooling_size, hidden=args.hidden_size,
                  history=args.hist, hist_len=args.hist_len,
                  num_class=args.num_act).to(device)
    agent_target = Agent(dim=args.pooling_size, hidden=args.hidden_size,
                  history=args.hist, hist_len=args.hist_len,
                  num_class=args.num_act).to(device)
    agent_target.load_state_dict(agent.state_dict())
    agent_target.eval()
    embeddingnet.eval()

    return embeddingnet, agent, agent_target

def feed_data(policy_net, target_net, embedding_net, data_loader, is_train):
    global n_iter
    if is_train:
        phase = 'Train'
        policy_net.train()
    else:
        phase = 'Test'
        policy_net.eval()

    total_pol_loss, total_rewards, total_length = [], [], []
    num_located = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        batch_size = data.shape[0]
        data, target = data.to(device), target.float()
        if phase == 'Train':
            optimizer_ag.zero_grad()

        pred_boxes = torch.tensor([[0, 0,
                                    83, 83]]).float().repeat(batch_size, 1)
        if args.hist == 1:
            action_history = torch.zeros(batch_size,
                                         args.hist_len, args.num_act).to(device)
        rois = torch.cat((torch.arange(0, batch_size).float().view(-1, 1),
                          pred_boxes), dim=1).to(device)
        actor = Actor(data, target, min_box_side=28)
        pol_losses, rewards_stps = 0, []
        boxes = []

        state = embedding_net.get_embedding(data, rois) # (bs, dim)
        state = torch.cat((state.view(batch_size,-1),
                           action_history.view(batch_size, -1)),1) \
                if args.hist == 1 else state
        for t in range(args.seq_len):
            actions, eps = select_action(state, policy_net, epoch, n_iter)
            # update action history
            if args.hist == 1:
                action_history[:, t%args.hist_len, :] = torch.nn.functional.one_hot(actions,
                                                        num_classes=args.num_act)
            rewards, new_boxes = actor.takeaction(actions)
            rois = torch.cat((torch.arange(0, state.shape[0]).float().view(-1, 1),
                              new_boxes), dim=1).float().to(device)
            next_state = embedding_net.get_embedding(data, rois)
            next_state = torch.cat((next_state.view(next_state.shape[0], -1),
                                    action_history.view(action_history.shape[0], -1)), 1) \
                         if args.hist == 1 else next_state
            done = torch.zeros_like(actions)
            done[actions == 5] = 1
            num_located += (rewards == 3).sum()

            # state (num_live, dim), actions (num_live, ),
            # next_state (num_live, dim), rewards (num_live, ), done (num_live,)
            memory.push_batch(state, actions, next_state, rewards, done)

            done_inds = (done == 0).cpu()
            state = next_state[done_inds]
            data = data[done_inds]
            actor.ious = actor.ious[done_inds]
            actor.targets = actor.targets[done_inds]
            actor.agent_windows = torch.from_numpy(actor.agent_windows)[done_inds].numpy()
            if args.hist == 1:
                action_history = action_history[done_inds]
            if state.shape[0] == 0:
                break

            # Perform one step of the optimization
            if phase == 'Train':
                loss = optimize_model(policy_net, target_net)
                pol_losses += loss.item()
                n_iter += 1
                #print('epoch ', epoch, ' batch ', batch_idx, ' n_iter ', n_iter)

            rewards_stps += [rewards]
            boxes.append(new_boxes)

            # Update the target network, copying all weights and biases in DQN
            if epoch % args.update_target_every == 0:
            #if n_iter % args.update_target_every == 0:
                target_net.load_state_dict(policy_net.state_dict())

            total_pol_loss.append(pol_losses)
            total_rewards.append(torch.mean(torch.cat(rewards_stps)))
        #break

    acc = 100. * num_located / len(data_loader.dataset)
    log_message = {
                   "{} Rewards".format(phase): np.mean(total_rewards),
                   #"{} Eps Length".format(phase): np.mean(total_length),
                   "{} Loc Acc".format(phase): acc,
                   'Train epsilon': eps}
    if phase == 'Train':
        current_lr = scheduler_ag.get_last_lr()[0]
        log_message.update({'learning rate': current_lr,
                            "{} Pol Loss".format(phase): np.mean(total_pol_loss)})
    wandb.log(log_message, step=epoch)

    return total_pol_loss, acc

def evaluate(policy_net, embedding_net, data_loader, upload=True):
    policy_net.eval()

    correct_loc = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        batch_size = data.shape[0]
        data, target = data.to(device), target.float()

        pred_boxes = torch.tensor([[0, 0,
                                    83, 83]]).float().repeat(batch_size, 1).to(device)
        rois = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                          pred_boxes), dim=1)
        actor = Actor(data, target, min_box_side=28)
        if args.hist == 1:
            action_history = torch.zeros(batch_size,
                                         args.hist_len, args.num_act).to(device)
        state = embedding_net.get_embedding(data, rois)
        state = torch.cat((state.view(batch_size, -1),
                           action_history.view(batch_size, -1)), 1) \
            if args.hist == 1 else state
        action_seq = torch.IntTensor(batch_size, args.seq_len).to(device)
        rewards_all = torch.zeros((batch_size, args.seq_len)).to(device)
        boxes = []
        for t in range(args.seq_len): # TODO stop action
            actions = policy_net(state).max(1)[1]
            rewards, new_boxes = actor.takeaction(actions)
            rois = torch.cat((torch.arange(0, state.shape[0]).float().view(-1, 1),
                              new_boxes), dim=1).float().to(device)
            state = embedding_net.get_embedding(data, rois)
            state = torch.cat((state.view(batch_size, -1),
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
    if upload:
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

def evaluate_gen_acc(upload=True):
    agent_path = os.path.join(saveroot, 'best.pth.tar')
    # load agent
    print('loading net ckpt from ', agent_path)
    ckpt = torch.load(agent_path)
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
            print('digit ', d, ' test ', len(testset))
            test_acc = evaluate(policy_net, embedding_net,
                                            data_loader=test_loader,
                                            upload=False)
            test_accs.append(test_acc)
            print(d, ' test_acc ', test_acc)
        print('mean acc ', np.mean(test_accs))
    if upload:
        wandb.run.summary.update({'Test Best Loc Acc': best_acc,
                              'Test Gen Acc': np.mean(test_accs)})

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
    scheduler_ag = torch.optim.lr_scheduler.StepLR(optimizer_ag,
                                                   step_size=args.step_ag, gamma=0.1)

    if args.evaluate == 1:
        DEBUG = False
        evaluate_gen_acc(upload=False)
        exit()

    memory = ReplayMemory(args.buffer_size)
    # The epsilon decay schedule
    epsilons = np.linspace(args.eps_start, args.eps_end, args.eps_decay_steps)

    best_acc = 0.
    save_model = False
    no_improve_epoch = 0
    n_iter = 0
    for epoch in range(args.epochs):
        train_log = feed_data(policy_net, target_net, embedding_net,
                              data_loader=train_loader, is_train=True)
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

    # evaluate gen acc
    DEBUG = False
    evaluate_gen_acc()

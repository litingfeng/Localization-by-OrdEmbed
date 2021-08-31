"""
baseline DQN model for clutter mnist localization
Created on 3/31/2021 2:42 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import os, wandb
import torch, copy
import random
import math
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from util.augmentations import Compose
from util.data_aug import Resize
from util.actions_tree_dqn import Actor
from opts import parser
from util.replay import Transition, ReplayMemory
from datasets.clutter_mnist_scale_rl import MNIST_CoLoc
from models.mnist_baseline_model import EmbeddingNet, Agent

DEBUG = True

def select_action(state, policy_net, epoch):
    sample = random.random()
    eps_threshold = epsilons[min(epoch, args.eps_decay_steps-1)]
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(args.num_act)]],
                            device=device, dtype=torch.long)

def init_dataloader():
    kwargs = {'num_workers': 8, 'pin_memory': True}

    train_transform = Compose([Resize(84)])
    test_transform = Compose([Resize(84)])
    trainset = MNIST_CoLoc(root='.', train=True, digit=4,
                           datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                           clutter=1, transform=train_transform
                           )
    testset = MNIST_CoLoc(root='.', train=False, digit=4,
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
                   pooling_mode=args.pooling_mode,).to(device)
    embeddingnet_dict = embeddingnet.state_dict()
    # Fiter out unneccessary keys
    mnist_pretrained_dict = torch.load('/research/cbim/vast/tl601/results/mnist/model.pth')
    filtered_dict = {}
    for k, v in mnist_pretrained_dict.items():
        if k.replace('feature.0', 'conv1') in embeddingnet_dict:
            filtered_dict[k.replace('feature.0', 'conv1')] = v
        elif k.replace('feature.3', 'conv2') in embeddingnet_dict:
            filtered_dict[k.replace('feature.3', 'conv2')] = v
        elif k in embeddingnet_dict:
            filtered_dict[k] = v
    embeddingnet_dict.update(filtered_dict)
    embeddingnet.load_state_dict(embeddingnet_dict)
    print('loaded pretained MNIST encoder')

    if args.freeze == 1:
        print('freezing embedding net')
        for name, value in embeddingnet.named_parameters():
            value.requires_grad = False

    agent = Agent(dim=1810, hidden=args.hidden_size, num_class=args.num_act).to(device)
    agent_target = Agent(dim=1810, hidden=args.hidden_size, num_class=args.num_act).to(device)
    agent_target.load_state_dict(agent.state_dict())
    agent_target.eval()

    return embeddingnet, agent, agent_target

def feed_data(policy_net, target_net, embedding_net, data_loader, is_train, n_iter):
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

        actions_seq = torch.ones(batch_size, args.seq_len) * 11
        for i in range(batch_size):
            pol_losses, rewards = 0, 0
            t = 0
            action = 0
            rois = torch.tensor([[0, 0, 0, 83., 83.]], dtype=torch.float32).to(device)
            img = data[i].unsqueeze(0)
            gt_box = target[i].view(-1, 4)
            # TODO, add action history
            state = embedding_net.get_embedding(img, rois)
            actor = Actor(img, gt_box.squeeze().numpy(), min_box_side=28)
            boxes = []
            while (action != 5) and (t < args.seq_len):
                action = select_action(state, policy_net, epoch)
                reward, new_box = actor.takeaction(action)
                if action == 5:
                    done = 1.
                else:
                    done = 0.

                if reward == 3:
                    num_located += 1

                rois = torch.cat((torch.tensor([0.]), torch.from_numpy(new_box))
                                 ).float().to(device)
                next_state = embedding_net.get_embedding(img, rois)
                memory.push(state, action, next_state, torch.tensor([reward]), torch.tensor([done]))
                state = next_state

                # Perform one step of the optimization
                if phase == 'Train':
                    loss = optimize_model(policy_net, target_net)
                    pol_losses += loss.item()
                    n_iter += 1

                rewards += reward
                actions_seq[i, t] = action
                boxes.append(new_box)
                t += 1

            if DEBUG and batch_idx % args.log_interval == 0:
                print("boxes ", np.stack(boxes)[-5:].astype(np.int8))
                print("actions ", actions_seq[i])
                print('target ', target[i])
                print('rewards ', rewards)

            # Update the target network, copying all weights and biases in DQN
            if epoch % args.update_target_every == 0:
            #if n_iter % args.update_target_every == 0:
                target_net.load_state_dict(policy_net.state_dict())

            total_pol_loss.append(pol_losses)
            total_rewards.append(rewards)
            total_length.append(t)

    acc = 100. * num_located / len(data_loader.dataset)
    log_message = {
                   "{} Rewards".format(phase): np.mean(total_rewards),
                   "{} Eps Length".format(phase): np.mean(total_length),
                   "{} Loc Acc".format(phase): acc}
    if phase == 'Train':
        current_lr = scheduler_ag.get_last_lr()[0]
        log_message.update({'learning rate': current_lr,
                            "{} Pol Loss".format(phase): np.mean(total_pol_loss)})
    wandb.log(log_message, step=epoch)

    return total_pol_loss, acc


def optimize_model(policy_net, target_net):
    if len(memory) < args.batch_size:
        return torch.zeros(1)
    transitions = memory.sample(args.batch_size)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    next_state_batch = torch.cat(batch.next_state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward).to(device)
    done_batch = torch.cat(batch.done).to(device)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Calculate q values and targets
    q_values_next = target_net(next_state_batch).max(1)[0].detach()
    expected_state_action_values = reward_batch + (1-done_batch) * args.gamma * q_values_next

    # Compute Huber loss
    #loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer_ag.zero_grad()
    loss.backward()
    # for param in policy_net.parameters():
    #     param.grad.data.clamp_(-1, 1)
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
    scheduler_ag = torch.optim.lr_scheduler.StepLR(optimizer_ag,
                                                   step_size=args.step_ag, gamma=0.1)

    memory = ReplayMemory(args.buffer_size)
    # The epsilon decay schedule
    epsilons = np.linspace(args.eps_start, args.eps_end, args.eps_decay_steps)

    best_acc = 0.
    save_model = False
    no_improve_epoch = 0
    n_iter = 0
    for epoch in range(args.epochs):
        train_log = feed_data(policy_net, target_net, embedding_net,
                              data_loader=train_loader, is_train=True, n_iter=n_iter)
        scheduler_ag.step()

        with torch.no_grad():
            test_loss, test_acc = feed_data(policy_net, target_net, embedding_net,
                                            data_loader=test_loader, is_train=False,
                                            n_iter=n_iter)

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

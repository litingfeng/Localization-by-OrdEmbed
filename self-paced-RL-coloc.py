"""
Use pretrained ordinal embedding to train RL in phase A.
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
from transform import MyBoxTransform
from util.augmentations import Compose
from util.data_aug import Resize
from torch.distributions import Categorical
from util.actions_tree_noscale import Actor
from util.utils import box_iou
from RL import get_policy_loss
from opts import parser
from datasets.clutter_mnist_rl import MNIST_CoLoc
from models.mnist_model import Net, EmbeddingNet, Agent

DEBUG = True
def init_dataloader():
    kwargs = {'num_workers': 8, 'pin_memory': True}

    train_transform = Compose([Resize(84)])
    test_transform = Compose([Resize(84)])
    trainset = MNIST_CoLoc(root='.', train=True, digit=4,
                           datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                           clutter=1, transform=train_transform,
                           target_transform=MyBoxTransform(num=1))
    testset = MNIST_CoLoc(root='.', train=False, digit=4,
                          datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                          clutter=1, transform=test_transform,
                          target_transform=MyBoxTransform(num=1))

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    print('total train image: ', len(train_loader.dataset), ' test image: ',
          len(test_loader.dataset))
    return train_loader, test_loader

def init_model():
    embedding_net = EmbeddingNet(norm=args.norm)
    net = Net(embedding_net, pooling_size=args.pooling_size,
              pooling_mode=args.pooling_mode).to(device)
    agent = Agent(dim=1810, num_class=args.num_act,
                  hidden_size=args.hidden_size).to(device)
    # if load pretrained encoder
    if args.pretrained != '':
        ckpt = torch.load(os.path.join('/research/cbim/vast/tl601/results/'
                          'selfpaced/ordinal/',
                          args.pretrained))
        net.load_state_dict(ckpt['state_dict'])
        print('loaded from {}\nckpt epoch {} acc {:.2f}%'.format(args.pretrained,
                                                    ckpt['epoch'], ckpt['acc']))
    if args.freeze == 1:
        for name, value in net.named_parameters():
            value.requires_grad = False

    return net, agent

def feed_data(model, agent, data_loader, is_train):
    if is_train:
        phase = 'Train'
        #model.train()
        agent.train()
    else:
        phase = 'Test'
        #model.eval()
        agent.eval()

    pol_losses, ent_losses, rewards = [], [], []
    correct = 0
    len_dataloader = len(data_loader)
    for batch_idx, (x0s, target, ti, pixi) in enumerate(data_loader):
        batch_size = x0s.shape[0]
        x0s, target, ti, pixi = x0s.to(device),  target.to(device).float(), \
                                ti.to(device), pixi.to(device)

        #optimizer.zero_grad()
        optimizer_ag.zero_grad()

        # rois
        rois_t0 = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                            target.to(device)), dim=1)
        actor = Actor(x0s, min_box_side=28)
        rewards_all = torch.zeros((batch_size, args.seq_len+1)).to(device)
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

        # pred_boxes = torch.tensor([[28, 28,
        #             56, 56]]).float().repeat(batch_size, 1)
        pred_boxes = ti
        rois = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                          pred_boxes), dim=1)
        # 1st forward classifier
        with torch.no_grad():
            embed_t = model(x0s, rois_t0)
            embed = model(x0s, rois)
        rewards_all[:, 0] = F.pairwise_distance(embed_t, embed)

        for t in range(args.seq_len):
            if args.hidden_size:
                h_t, logits, actions, _ = agent(embed, h_t_prev=h_t)
            else:
                logits, actions, _ = agent(embed)
            state, pred_boxes = actor.takeaction(actions)
            rois = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                              pred_boxes.to(device)), dim=1)
            with torch.no_grad():
                embed = model(x0s, rois)
            all_embed.append(embed)
            #rewards_all[:, t+1] = F.pairwise_distance(embed_t, embed)
            action_seq[:, t] = actions.view(1, -1)
            logits_seq.append(logits)
            boxes.append(pred_boxes)

        rewards_trns = F.pairwise_distance(embed_t.repeat(args.seq_len, 1),
                                           torch.cat(all_embed))
        rewards_all[:, 1:] = rewards_trns.view(-1, batch_size).permute(1,0)

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
        q, score, critic_loss, adv = get_policy_loss(rewards_deltas, batch_size, gamma=args.gamma,
                                                     logits_seq=logits_seq, seq_len=args.seq_len,
                                                     coo_actions=coo_actions, values=values)


        m = Categorical(logits=logits)
        entropy_loss = m.entropy().mean()

        loss = q
        if phase == 'Train':
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 10)
            optimizer_ag.step()

        # calculate accuracy
        iou = torch.diagonal(box_iou(pred_boxes.float(),
                                     target.cpu()), 0)
        correct += (iou >= 0.5).sum()

        pol_losses.append(q.item())
        ent_losses.append(entropy_loss.item())
        rewards.append(torch.mean(rewards_deltas).item())

        if DEBUG:
            print('action_seq ', action_seq[:5])
            print('rewards_all ', -rewards_all[:5])
            print('rewards_deltas ', rewards_deltas[:5])
            print('adv ', adv[:5])
            print('pred_boxes ', pred_boxes[:5])
            print('ti ', ti[:5])
            print('target_boxes ', target[:5].int())

        if batch_idx % args.log_interval == 0:
            if batch_idx == 0:
                print('\n')
            print('{} Epoch: {} [{}/{} ({:.0f}%)]\tReward: {:.4f}'.format(
                phase, epoch, batch_idx * batch_size,
                len(data_loader.dataset),
                100. * batch_idx / len_dataloader, np.mean(rewards)))

    total_pol_loss = np.mean(pol_losses)
    acc = 100. * correct / len(data_loader.dataset)
    log_message = {"{} Pol Loss".format(phase): total_pol_loss,
                   '{} Rewards'.format(phase): np.mean(rewards),
                   '{} Acc'.format(phase): acc,
                   '{} Ent Loss'.format(phase): np.mean(ent_losses)
                   }

    if phase == 'Train':
        current_lr = scheduler.get_last_lr()[0]
        log_message.update({'learning rate': current_lr})
    wandb.log(log_message, step=epoch)

    return total_pol_loss, acc

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
        optimizer_ag = torch.optim.SGD(agent.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        optimizer_ag = torch.optim.Adam(agent.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.1)
    scheduler_ag = torch.optim.lr_scheduler.StepLR(optimizer_ag, step_size=args.step, gamma=0.1)

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
                    'acc': test_acc
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

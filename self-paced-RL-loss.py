"""
Randomly rotate range(0, 181, 15) for phase A.
In each epoch:
    update proto with confident
    in each iteration:
    1. forward and train classifier
    2. train agent to transform less confident samples

Created on 2/18/2021 7:18 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import os, wandb
import torch, copy

import torch.nn as nn
import torch.nn.functional as F
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from actions2 import Actor
from util import cosine_simililarity, sample
from torchvision import datasets, transforms
from RL import get_policy_loss
from opts import parser
from datasets.cifar_seed import myCIFAR
from transform import MyRotationTransform
from models.cifar_model import Net, EmbeddingNet, Agent

DEBUG = True
def init_dataloader():
    train_set = myCIFAR(root='.', datapath='.', train=True,
                      transform=MyRotationTransform(angles=np.array(range(0,181, args.angle_step)),
                                                    num=1))

    test_set = myCIFAR(root='.', datapath='.',
                       train=False,
                       transform=MyRotationTransform(angles=np.array(range(0,181, args.angle_step)),
                                                    num=1))
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size,
                                               shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=args.batch_size,
                                              shuffle=False, num_workers=8)
    print('total images of 4 & 1: ', len(train_set), ' test: ', len(test_set))

    return train_loader, test_loader

def init_model():
    embedding_net = EmbeddingNet()
    net = Net(embedding_net, 2).to(device)
    agent = Agent(dim=5, num_action=2, hidden_size=args.hidden_size).to(device)
    # if load pretrained classifier
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
        model.train()
        agent.train()
    else:
        phase = 'Test'
        model.eval()
        agent.eval()

    thresh = args.thresh
    pol_losses, rewards, mses, loss_is = [], [], [], []
    correct, correct_rnd = 0, 0
    len_dataloader = len(data_loader)
    for batch_idx, (ai, xis, target) in enumerate(data_loader):
        batch_size = xis.shape[0]
        ai, xis, target = ai.to(device),xis.to(device), \
                          target.to(device)

        #optimizer.zero_grad()
        optimizer_ag.zero_grad()

        # take actions
        actor = Actor()
        loss_all = torch.zeros((batch_size, (args.seq_len + 1))).to(device)
        action_seq = torch.IntTensor(batch_size, args.seq_len).to(device)
        logits_seq = []
        if args.hidden_size:
            h_t = torch.zeros(
                batch_size,
                args.hidden_size,
                dtype=torch.float,
                device=device,
                requires_grad=True,
            )
        # 1st forward classifier
        with torch.no_grad():
            feature_i, output_i = model(xis)
        loss_i = criterion(output_i, target) # (bs,)
        loss_all[:, 0] = loss_i

        for t in range(args.seq_len):
            if args.hidden_size:
                h_t, logits, actions = agent(feature_i, h_t_prev=h_t)
            else:
                logits, actions = agent(feature_i) # actions: (bs, 1)
            xis = actor.takeaction(xis, actions)
            with torch.no_grad():
                feature_i, output_i = model(xis)
            loss_i = criterion(output_i, target)

            loss_all[:, t+1] = loss_i
            action_seq[:, t] = actions.view(1, -1)
            logits_seq.append(logits)

        if args.sparse == 0:
            # incremental reward
            loss_deltas = -loss_all[:, 1:] - (-loss_all[:, :-1])
            if args.sign == 1:
                loss_deltas = torch.sign(loss_deltas)
        else:
            # sparse reward
            loss_deltas = - (loss_all[:, -1] - loss_all[:, 0])
            loss_deltas = loss_deltas.unsqueeze(1).repeat(1, args.seq_len)
            if args.sign == 1:
                loss_deltas = torch.sign(loss_deltas)

        # Get the policy loss
        coord = product(range(action_seq.size(0)), range(action_seq.size(1)))
        coo_actions = [[k, m, action_seq[k, m]] for k, m in coord]

        logits_seq = torch.stack(tuple(logits_seq), dim=0)
        logits_seq = logits_seq.permute(1, 0, 2).to(device) # (bs, T, dim)

        values = None
        q, score, critic_loss, adv = get_policy_loss(loss_deltas, batch_size, gamma=args.gamma,
                                                     logits_seq=logits_seq, seq_len=args.seq_len,
                                                     coo_actions=coo_actions, values=values)

        loss = q
        if phase == 'Train':
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 10)
            optimizer_ag.step()

        # calculate pred angle
        angles = action_seq.sum(1) * float(args.angle_step)
        correct += (angles == ai).sum().item()
        angles_rnd = torch.randint(0,2,size=(batch_size, args.seq_len)).sum(1).to(device) \
                                   * float(args.angle_step)
        correct_rnd += (angles_rnd == ai).sum().item()
        mse = F.mse_loss(angles.unsqueeze(1), ai.unsqueeze(1))
        pol_losses.append(q.item())
        rewards.append(torch.mean(loss_deltas).item())
        mses.append(mse.item())
        loss_is.append(torch.mean(loss_all[:,0]).item())

        if DEBUG:
            print('action_seq ', action_seq[:5])
            print('loss_all ', -loss_all[:5])
            print('loss_all_delta ', loss_deltas[:5])
            print('adv ', adv[:5])
            print('ai ', ai[:5])
            print('pred ', angles[:5])

        if batch_idx % args.log_interval == 0:
            if batch_idx == 0:
                print('\n')
            print('{} Epoch: {} [{}/{} ({:.0f}%)]\tReward: {:.4f}'.format(
                phase, epoch, batch_idx * batch_size,
                len(data_loader.dataset),
                100. * batch_idx / len_dataloader, np.mean(rewards)))

    total_pol_loss = np.mean(pol_losses)
    acc = 100. * correct / len(data_loader.dataset)
    acc_rnd = 100. * correct_rnd / len(data_loader.dataset)
    log_message = {"{} Pol Loss".format(phase): total_pol_loss,
                   "{} Cls Loss".format(phase): np.mean(loss_is),
                   '{} Rewards'.format(phase): np.mean(rewards),
                   '{} MSE'.format(phase): np.mean(mses),
                   '{} Acc'.format(phase): acc,
                   '{} Acc_rand'.format(phase): acc_rnd
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

    criterion = nn.CrossEntropyLoss(reduction='none')
    triplet_loss = nn.TripletMarginLoss(margin=args.margin, p=2)
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
        optimizer_ag = torch.optim.SGD(agent.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        optimizer_ag = torch.optim.Adam(agent.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.1)
    scheduler_ag = torch.optim.lr_scheduler.StepLR(optimizer_ag, step_size=args.step, gamma=0.1)

    criterion = nn.CrossEntropyLoss(reduction='none')
    criterion_ag = nn.CrossEntropyLoss(reduction='none')

    best_acc = 0.
    save_model = False
    no_improve_epoch = 0
    for epoch in range(args.epochs):
        train_log = feed_data(net, agent, data_loader=train_loader, is_train=True)
        scheduler.step()

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
                    'state_dict': net.state_dict(),
                    'acc': test_acc
                }, os.path.join(saveroot, 'best.pth.tar'))

            # break training
            if no_improve_epoch > args.patience:
                print('stop training...')
                break

    torch.save({
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'acc': test_acc
    }, os.path.join(saveroot, 'last.pth.tar'))

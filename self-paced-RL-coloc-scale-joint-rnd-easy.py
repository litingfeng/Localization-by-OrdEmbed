"""
For rnd model, no normalize
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
from transform import MyBoxScaleTransform
from util.augmentations import Compose
from util.data_aug import Resize
from torch.distributions import Categorical
from util.actions_tree_bk import Actor
from util.utils import box_iou, RunningMeanStd, RewardForwardFilter
from RL_entropy import get_policy_loss, _build_discounts_matrix
from opts import parser
from datasets.clutter_mnist_scale_rl import MNIST_CoLoc
from models.mnist_scale_model import Net, Agent
from models.rnd_model import RNDModel

DEBUG = True
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
    agent = Agent(dim=512, num_class=args.num_act,
                  hidden_size=args.hidden_size).to(device)
    rnd_model = RNDModel(512, device).to(device) #TODO embed or roi as input
    # if load pretrained encoder
    if args.pretrained != '':
        ckpt = torch.load(os.path.join('/research/cbim/vast/tl601/results/'
                          'selfpaced/ordinal/',
                          args.pretrained))
        net.load_state_dict(ckpt['state_dict'])
        print('loaded from {}\nckpt epoch {} acc {:.2f}%'.format(args.pretrained,
                                                    ckpt['epoch'], ckpt['acc']))
    assert (args.freeze == 0)
    if args.freeze == 1:
        for name, value in net.named_parameters():
            value.requires_grad = False

    return net, agent, rnd_model

def feed_data(model, agent, rnd, data_loader, is_train):
    if is_train:
        phase = 'Train'
        model.train()
        agent.train()
        rnd.train()
    else:
        phase = 'Test'
        model.eval()
        agent.eval()
        rnd.eval()

    iouis, ioujs = [], []
    pol_losses, ent_losses, ord_losses, rewards, int_losses = [], [], [], [], []
    rewards_int, rewards_total = [], []
    correct_loc, correct_ord_inst, correct_ord_proto = 0, 0, 0
    len_dataloader = len(data_loader)
    for batch_idx, (x0s, target) in enumerate(data_loader):
        batch_size = x0s.shape[0]
        x0s, target = x0s.to(device), target.to(device).float()


        optimizer.zero_grad()
        optimizer_ag.zero_grad()

        '''train agent'''
        rois_t0 = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                             target), dim=1)
        actor = Actor(x0s, min_box_side=28)
        rewards_all = torch.zeros((batch_size, args.seq_len+1)).to(device)
        action_seq = torch.IntTensor(batch_size, args.seq_len).to(device)
        int_reward = torch.zeros((batch_size, args.seq_len)).to(device)
        logits_seq, boxes, intr_loss = [], [], []
        all_embed = []
        if args.hidden_size:
            h_t = torch.zeros(
                batch_size,
                args.hidden_size,
                dtype=torch.float,
                device=device,
                requires_grad=True,
            )

        pred_boxes = torch.tensor([[0, 0,
                    83, 83]]).float().repeat(batch_size, 1).to(device)
        boxes.append(pred_boxes.cpu())
        rois = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                          pred_boxes), dim=1)
        # 1st forward classifier
        embed_t = model(x0s, rois_t0)
        with torch.no_grad():
            embed = model.get_roi_embedding(rois)
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
                embed = model.get_roi_embedding(rois)
            all_embed.append(embed)

            # calculate intrinsic reward TODO clip value
            predict_next_state_feature, target_next_state_feature = rnd(embed.detach())
            intrinsic_reward = (predict_next_state_feature -
                                target_next_state_feature).pow(2).sum(1) / 2

            forward_loss = F.mse_loss(predict_next_state_feature,
                                      target_next_state_feature.detach())
            # Proportion of exp used for predictor update
            # mask = torch.rand(len(forward_loss)).to(device)
            # mask = (mask < 0.25).type(torch.FloatTensor).to(device)
            # forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(device))
            intr_loss.append(forward_loss)
            action_seq[:, t] = actions.view(1, -1)
            logits_seq.append(logits)
            boxes.append(pred_boxes)
            int_reward[:, t] = intrinsic_reward.view(1, -1) # reverse???

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
        #TODO, deal with stop action, modify rewards
        q, score, critic_loss, adv, adv_int = get_policy_loss(rewards_deltas, batch_size, gamma=args.gamma,
                                                     logits_seq=logits_seq, seq_len=args.seq_len,
                                                     coo_actions=coo_actions, values=values,
                                                     int_reward=int_reward)

        m = Categorical(logits=logits_seq)
        loss_ent = m.entropy() # (bs, T)

        loss_int = torch.tensor(intr_loss).sum()
        loss = (q - loss_ent * args.lamb_ent).sum() + \
               args.lamb_int * loss_int

        if phase == 'Train':
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(agent.parameters(), 10)
            optimizer_ag.step()

        '''train embedding net using boxes from agent'''
        boxes = torch.stack(boxes).permute(1,0,2).to(device) # (bs, T+1, 4)
        inds = [np.random.choice(args.seq_len+1,
                                 size=2, replace=False) for _ in range(batch_size)]
        smp_boxes = torch.zeros(batch_size, 2, 4).to(device)
        for i in range(batch_size):
            smp_boxes[i] = boxes[i, inds[i], :]

        inds = torch.arange(0, batch_size).float().view(-1, 1).repeat(1,
                          2).to(device)
        rois = torch.cat((inds.view(-1, 1),
                          smp_boxes.view(-1,4)), dim=1) # (bs*2, 5)

        iou = [box_iou(smp_boxes[i], target[i].view(-1, 4))
                for i in range(batch_size)]
        iou = torch.stack(iou).squeeze() # (bs, 2)
        feature = model(x0s, rois.detach()).view(batch_size, 2, -1)  # (bs, 2, 1024)

        # compute loss order
        inds = torch.max(iou, 1).indices
        p = feature[torch.arange(0, batch_size), inds, :]
        n = feature[torch.arange(0, batch_size), 1 - inds, :]
        proto = torch.mean(embed_t, dim=0).view(1, -1).repeat(batch_size, 1)
        # loss = triplet_loss(proto, p, n)
        loss_ord = triplet_loss(embed_t, p, n)

        # compute accuracy
        d_a0a1 = F.pairwise_distance(embed_t, feature[:,0,:])
        d_a0a2 = F.pairwise_distance(embed_t, feature[:,1,:])
        sub_pix = iou[:,1] - iou[:,0]
        sub_d = d_a0a1 - d_a0a2
        mask = (torch.sign(sub_d) == torch.sign(sub_pix))
        correct_ord_inst += mask.sum().item()

        d_a0a1 = F.pairwise_distance(proto, feature[:,0,:])
        d_a0a2 = F.pairwise_distance(proto, feature[:,1,:])
        sub_d = d_a0a1 - d_a0a2
        mask = (torch.sign(sub_d) == torch.sign(sub_pix))
        correct_ord_proto += mask.sum().item()

        # update model
        if phase == 'Train':
            loss_ord.backward()
            optimizer.step()

        # calculate accuracy
        final_iou = torch.diagonal(box_iou(pred_boxes.float(),
                                     target.cpu()), 0)
        correct_loc += (final_iou >= 0.5).sum()

        pol_losses.append(q.sum().item())
        ent_losses.append(loss_ent.sum().item())
        ord_losses.append(loss_ord.item())
        int_losses.append(loss_int.item())
        iouis.append(iou[:,0])
        ioujs.append(iou[:,1])
        rewards.append(torch.mean(rewards_deltas).item())
        rewards_int.append(torch.mean(int_reward).item())
        rewards_total.append(torch.mean(rewards_deltas+int_reward).item())

        if DEBUG:
            print('action_seq ', action_seq[:5])
            print('rewards_all ', -rewards_all[:5])
            print('rewards_deltas ', rewards_deltas[:5])
            print('adv ', adv[:5])
            print('adv_int ', adv_int[:5])
            print('pred_boxes ', pred_boxes[:5])
            print('target_boxes ', target[:5].int())

        if batch_idx % args.log_interval == 0:
            if batch_idx == 0:
                print('\n')
            print('{} Epoch: {} [{}/{} ({:.0f}%)]\tReward: {:.4f}'.format(
                phase, epoch, batch_idx * batch_size,
                len(data_loader.dataset),
                100. * batch_idx / len_dataloader, np.mean(rewards)))

    total_pol_loss = np.mean(pol_losses)
    acc_loc = 100. * correct_loc / len(data_loader.dataset)
    acc_inst = 100. * correct_ord_inst / len(data_loader.dataset)
    acc_proto = 100. * correct_ord_proto / len(data_loader.dataset)

    iouis = torch.cat(iouis).cpu().numpy()
    ioujs = torch.cat(ioujs).cpu().numpy()
    f, ax = plt.subplots(2, 1)
    ax[0].hist(iouis, label='iouis')
    ax[1].hist(ioujs, label='ioujs')
    ax[0].legend()
    ax[1].legend()


    log_message = {"{} Pol Loss".format(phase): total_pol_loss,
                   "{} Ord Loss".format(phase): np.mean(ord_losses),
                   '{} Rewards'.format(phase): np.mean(rewards_total),
                   '{} Int Rewards'.format(phase): np.mean(rewards_int),
                   '{} Ext Rewards'.format(phase): np.mean(rewards),
                   '{} Loc Acc'.format(phase): acc_loc,
                   '{} Ent Loss'.format(phase): np.mean(ent_losses),
                   '{} Int Loss'.format(phase): np.mean(int_losses),
                   '{} Ord Acc Inst'.format(phase): acc_inst,
                   '{} Ord Acc Proto'.format(phase): acc_proto,
                   '{} IoU Distribution'.format(phase): wandb.Image(plt)
                   }

    if phase == 'Train':
        current_lr = scheduler_ag.get_last_lr()[0]
        log_message.update({'learning rate': current_lr})
    wandb.log(log_message, step=epoch)

    return total_pol_loss, acc_loc

if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ['WANDB_NAME'] = args.savename
    wandb.init(project="selfpaced")
    saveroot = os.path.join('/research/cbim/vast/tl601/results/selfpaced/ordinal',
                            args.savename)
    os.makedirs(saveroot, exist_ok=True)

    train_loader, test_loader = init_dataloader()

    net, agent, rnd_model = init_model()

    wandb.watch(rnd_model, log_freq=10)
    wandb.watch(agent, log_freq=10)
    wandb.config.update(args)

    triplet_loss = nn.TripletMarginLoss(margin=args.margin, p=2)
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    if args.optimizer_ag == 'SGD':
        optimizer_ag = torch.optim.SGD(list(agent.parameters()) +
                                    list(rnd_model.predictor.parameters()), lr=args.lr_ag, momentum=0.9)
    elif args.optimizer_ag == 'Adam':
        optimizer_ag = torch.optim.Adam(list(agent.parameters()) +
                                    list(rnd_model.predictor.parameters()), lr=args.lr_ag)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=args.steps, gamma=0.1)
    scheduler_ag = torch.optim.lr_scheduler.StepLR(optimizer_ag,
                                                   step_size=args.step_ag, gamma=0.1)

    if args.evaluate == 1:
        net_path = os.path.join(saveroot, 'best_net.pth.tar')
        agent_path = os.path.join(saveroot, 'best_agent.pth.tar')
        # load net
        print('loading net ckpt from ', net_path)
        ckpt = torch.load(net_path)
        state_dict = ckpt['state_dict']
        epoch = ckpt['epoch']
        best_acc = ckpt['acc']
        net.load_state_dict(state_dict)
        print(("=> loaded net checkpoint epoch {} {}".format(epoch, best_acc)))
        # load agent
        print('loading net ckpt from ', agent_path)
        ckpt = torch.load(agent_path)
        state_dict = ckpt['state_dict']
        epoch = ckpt['epoch']
        best_acc = ckpt['acc']
        agent.load_state_dict(state_dict)
        print(("=> loaded agent checkpoint epoch {} {}".format(epoch, best_acc)))
        test_loss, test_acc = feed_data(net, agent,  data_loader=test_loader, is_train=False)
        print('test_acc ', test_acc)
        exit()

    best_acc = 0.
    save_model = False
    no_improve_epoch = 0
    for epoch in range(args.epochs):
        train_log = feed_data(net, agent, rnd_model, data_loader=train_loader, is_train=True)
        scheduler_ag.step()
        scheduler.step()

        with torch.no_grad():
            test_loss, test_acc = feed_data(net, agent, rnd_model, data_loader=test_loader, is_train=False)

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
                }, os.path.join(saveroot, 'best_agent.pth.tar'))
                torch.save({
                    'epoch': epoch,
                    'state_dict': net.state_dict(),
                    'acc': test_acc
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
        'state_dict': net.state_dict(),
        'acc': test_acc
    }, os.path.join(saveroot, 'last_net.pth.tar'))

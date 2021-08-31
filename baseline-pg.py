"""
Using autoencoder or AE+Ord + IoU reward + policy gradient
Created on 4/14/2021 11:18 AM

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
from util.data_aug import Resize, RandomTranslate, RandomHorizontalFlip, RandomShear
from torch.distributions import Categorical
from util.actions_tree_dqn_batch import Actor
from util.utils import box_iou
from RL_entropy import get_policy_loss
from opts import parser
from datasets.clutter_mnist_scale_rl import MNIST_CoLoc
from models.mnist_baseline_ae_model import EmbeddingNet
from models.mnist_scale_model import Agent_ae, Net_ae, Net_ae_proj

DEBUG = True
def init_dataloader():
    kwargs = {'num_workers': 8, 'pin_memory': True}

    # train_transform = Compose([RandomHorizontalFlip(),
    #                            # RandomShear(),
    #                            RandomTranslate(),
    #                            Resize(84)])
    train_transform = Compose([Resize(84)])
    test_transform = Compose([Resize(84)])
    trainset = MNIST_CoLoc(root='.', train=True, digit=args.digit,
                           sample_size=args.sample_size, bg_name=args.bg_name,
                           datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                           clutter=1, transform=train_transform,
                           )
    testset = MNIST_CoLoc(root='.', train=False, digit=args.digit,bg_name=args.bg_name,
                          datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                          clutter=1, transform=test_transform,
                          )

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    print('total train image: ', len(train_loader.dataset), ' test image: ',
          len(test_loader.dataset))
    return train_loader, test_loader

def init_model():
    embeddingnet = Net_ae(pooling_size=args.pooling_size,
                                pooling_mode=args.pooling_mode).to(device)
    embeddingnet_dict = embeddingnet.state_dict()
    # Fiter out unneccessary keys
    model_path = '/research/cbim/vast/tl601/results/' \
                 'mnist/{}'.format(args.pretrained)
    mnist_pretrained_dict = torch.load(model_path)

    filtered_dict = {}
    for k, v in mnist_pretrained_dict.items():
        if k in embeddingnet_dict:
            filtered_dict[k] = v
    embeddingnet_dict.update(filtered_dict)
    embeddingnet.load_state_dict(embeddingnet_dict)
    print('loaded pretained MNIST encoder ', model_path)
    # for ordinal+ae
    # embeddingnet = Net_ae_proj(pooling_size=args.pooling_size,
    #              pooling_mode=args.pooling_mode).to(device)
    # if args.pretrained != '':
    #     ckpt = torch.load(os.path.join('/research/cbim/vast/tl601/results/'
    #                                    'mnist/', args.pretrained))
    #     if 'state_dict' in ckpt.keys():
    #         mnist_pretrained_dict = ckpt['state_dict']
    #     else:
    #         mnist_pretrained_dict = ckpt
    #     embeddingnet_dict = embeddingnet.state_dict()
    #
    #     filtered_dict = {}
    #     for k, v in mnist_pretrained_dict.items():
    #         if k in embeddingnet_dict:
    #             filtered_dict[k] = v
    #     embeddingnet_dict.update(filtered_dict)
    #     embeddingnet.load_state_dict(embeddingnet_dict)
    #     if 'state_dict' in ckpt.keys():
    #         print('loaded from {}\nckpt epoch {} acc {:.2f}%'.format(args.pretrained,
    #                                 ckpt['epoch'], ckpt['acc']))
    #     else:
    #         print('loaded pretained MNIST encoder ', args.pretrained)

    agent = Agent_ae(rnn=args.rnn, poolsize=args.pooling_size,
                     num_class=args.num_act,
                     hidden_size=args.hidden_size).to(device)
    assert (args.freeze == 1)
    if args.freeze == 1:
        for name, value in embeddingnet.named_parameters():
            value.requires_grad = False
        embeddingnet.eval()

    return embeddingnet, agent

def feed_data(model, agent, data_loader, is_train, upload=True):
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
    for batch_idx, (data, target) in enumerate(data_loader):
        batch_size = data.shape[0]
        data, target = data.to(device),  target.float()
        if phase == 'Train':
            #optimizer.zero_grad()
            optimizer_ag.zero_grad()

        actor = Actor(data, target, min_box_side=28, sign=args.sign)
        rewards_all = torch.zeros((batch_size, args.seq_len))
        action_seq = torch.IntTensor(batch_size, args.seq_len).to(device)
        logits_seq, boxes = [], []
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
            state = model(data, rois)

        for t in range(args.seq_len):
            if args.rnn == 1:
                h_t, logits, actions = agent(state, h_t_prev=h_t)
            else:
                logits, actions = agent(state)
            reward, pred_boxes = actor.takeaction(actions)
            rois = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                              pred_boxes.to(device)), dim=1)
            with torch.no_grad():
                state = model.get_roi_embedding(rois)

            action_seq[:, t] = actions.view(1, -1)
            logits_seq.append(logits)
            rewards_all[:, t] = reward
            boxes.append(pred_boxes)

        # Get the policy loss
        coord = product(range(action_seq.size(0)), range(action_seq.size(1)))
        coo_actions = [[k, m, action_seq[k, m]] for k, m in coord]

        logits_seq = torch.stack(tuple(logits_seq), dim=0)
        logits_seq = logits_seq.permute(1, 0, 2).to(device) # (bs, T, dim)

        #TODO, deal with stop action, modify rewards
        q, score, critic_loss, adv = get_policy_loss(rewards_all.to(device), batch_size, gamma=args.gamma,
                                                     logits_seq=logits_seq, seq_len=args.seq_len,
                                                     coo_actions=coo_actions)

        m = Categorical(logits=logits_seq)
        loss_ent = m.entropy() # (bs, T)

        loss = (q - loss_ent * args.lamb_ent).sum()

        if phase == 'Train':
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(agent.parameters(), 10)
            optimizer_ag.step()

        # calculate accuracy
        iou = torch.diagonal(box_iou(pred_boxes.float(),
                                     target), 0)
        correct += (iou >= 0.5).sum()

        pol_losses.append(q.sum().item())
        ent_losses.append(loss_ent.sum().item())
        rewards.append(torch.mean(rewards_all).item())

        if DEBUG:
            print('action_seq ', action_seq[:5])
            print('rewards_all ', rewards_all[:5])
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

def evaluate_gen_acc(upload=True):
    agent_path = os.path.join(saveroot, 'best.pth.tar')
    # load agent
    print('loading net ckpt from ', agent_path)
    ckpt = torch.load(agent_path)
    state_dict = ckpt['state_dict']
    epoch = ckpt['epoch']
    best_acc = ckpt['acc']
    agent.load_state_dict(state_dict)
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
            test_loss, test_acc = feed_data(net, agent,
                                            data_loader=test_loader, is_train=False,
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
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=args.steps, gamma=0.1)
    scheduler_ag = torch.optim.lr_scheduler.StepLR(optimizer_ag,
                                                   step_size=args.step_ag, gamma=0.1)

    if args.evaluate == 1:
        DEBUG = False
        evaluate_gen_acc(upload=False)
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

    # evaluate gen acc
    DEBUG = False
    evaluate_gen_acc()

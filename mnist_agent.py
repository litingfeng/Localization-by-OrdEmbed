# -*- coding: utf-8 -*-
# @Time : 9/8/21 3:44 PM
# @Author : Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.

import os, wandb, torch
import numpy as np
from opts import parser
import torch.nn.functional as F
from itertools import product
from util.augmentations import Compose
from torch.distributions import Categorical
from util.data_aug import Resize
from datasets.clutter_mnist import MNIST_Corrupted
from models.mnist_model import AutoencoderProj_ag, Agent
from util.actions import Actor
from util.utils import box_iou
from RL import get_policy_loss

IMG_SIZE = 84
DEBUG = False

def init_dataloader():
    kwargs = {'num_workers': 8, 'pin_memory': True}
    if args.dataset == 'mnist':
    transform = Compose([Resize(84)])
        trainset = MNIST_Corrupted(root='.', train=True, digit=args.digit,
                               bg_name=args.bg_name,
                               sample_size=args.sample_size,
                               datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist_github',
                                transform=transform)
        testset = MNIST_Corrupted(root='.', train=False, digit=args.digit,
                              bg_name=args.bg_name,
                              datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist_github',
                              transform=transform)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    print('total train image: ', len(train_loader.dataset), ' test image: ',
          len(test_loader.dataset))
    return train_loader, test_loader

def init_model():
    net = AutoencoderProj_ag(channel=1, pooling_size=args.pooling_size,
                             dim=args.dim, pooling_mode=args.pooling_mode).to(device)
    agent = Agent(rnn=args.rnn, poolsize=args.pooling_size,
                  num_class=args.num_act, hidden_size=args.hidden_size).to(device)
    '''load pretrained net'''
    if args.pretrained != '':
        ckpt = torch.load(os.path.join('/research/cbim/vast/tl601/results/'
                                       'mnist/', args.pretrained))
        mnist_pretrained_dict = ckpt['state_dict']
        embeddingnet_dict = net.state_dict()

        filtered_dict = {}
        for k, v in mnist_pretrained_dict.items():
            if k in embeddingnet_dict:
                print('load ', k)
                filtered_dict[k] = v
        embeddingnet_dict.update(filtered_dict)
        net.load_state_dict(embeddingnet_dict)
        print('loaded from {}\nckpt epoch {} acc {:.2f}%'.format(args.pretrained,
                                                                 ckpt['epoch'], ckpt['acc_inst']))

    if args.freeze:
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
    for batch_idx, (data, target) in enumerate(data_loader):
        batch_size = data.shape[0]
        data, target = data.to(device),  target.float()
        optimizer_ag.zero_grad()

        '''split to support set and target set '''
        num = batch_size % args.samples_per_class
        if num != 0:
            data, target = data[:-num], target[:-num]
            batch_size = data.shape[0]

        num_group = batch_size // args.samples_per_class
        rnd_inds = np.random.permutation(num_group)
        if num_group == 0:
            continue
        count += batch_size

        '''initialize'''
        rois_gt = torch.cat((torch.arange(0, batch_size).float().view(-1, 1),
                             target), dim=1).to(device)
        actor = Actor(data.shape[2:], data.shape[0], min_box_side=args.min_box_side)
        rewards_all = torch.zeros((batch_size, args.seq_len+1)).to(device)
        action_seq = torch.IntTensor(batch_size, args.seq_len).to(device)
        logits_seq, boxes = [], []
        if args.hidden_size:
            h_t = torch.zeros(
                batch_size,
                args.hidden_size,
                dtype=torch.float,
                device=device,
                requires_grad=True,
            )
        # start from the whole image
        pred_boxes = torch.tensor([[0, 0, IMG_SIZE-1, IMG_SIZE-1]]).float().repeat(batch_size, 1).to(device)
        rois = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                          pred_boxes), dim=1)

        '''forward embedding net'''
        with torch.no_grad():
            embed_gt, top_feat_gt = model(data, rois_gt)
            embed, top_feat       = model.get_roi_embedding(rois)

        '''compute reward of the first step '''
        if 'proto' in args.train_mode:
            proto = top_feat_gt.view(-1, args.samples_per_class, args.dim).mean(1) # (num_group, 512)
            if args.train_mode == 'shuffle_proto':
                proto = proto[rnd_inds]
            proto = proto.unsqueeze(1).repeat(1, args.samples_per_class, 1).view(-1, args.dim)
            rewards_all[:, 0] = F.pairwise_distance(proto, top_feat)
        else:
            if args.train_mode == 'shuffle_self':
                top_feat_gt = top_feat_gt.view(-1, args.samples_per_class, args.dim)[rnd_inds].view(-1, args.dim)
            rewards_all[:, 0] = F.pairwise_distance(top_feat_gt, top_feat)

        '''forward agent'''
        for t in range(args.seq_len):
            if args.hidden_size:
                h_t, logits, actions = agent(embed, h_t_prev=h_t)
            else:
                logits, actions = agent(embed)
            pred_boxes = actor.take_action(actions)
            rois = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                              pred_boxes.to(device)), dim=1)
            with torch.no_grad():
                embed, top_feat = model.get_roi_embedding(rois)
            if 'proto' in args.train_mode:
                rewards_all[:, t + 1] = F.pairwise_distance(proto, top_feat)
            else:
                rewards_all[:, t + 1] = F.pairwise_distance(top_feat_gt, top_feat)
            action_seq[:, t] = actions.view(1, -1)
            logits_seq.append(logits)
            boxes.append(pred_boxes)

        '''incremental reward'''
        rewards_deltas = -rewards_all[:, 1:] - (-rewards_all[:, :-1])
        if args.sign:
            rewards_deltas = torch.sign(rewards_deltas)

        '''compute policy loss'''
        coord = product(range(action_seq.size(0)), range(action_seq.size(1)))
        coo_actions = [[k, m, action_seq[k, m]] for k, m in coord]

        logits_seq = torch.stack(tuple(logits_seq), dim=0)
        logits_seq = logits_seq.permute(1, 0, 2).to(device) # (bs, T, dim)

        #TODO, deal with stop action, modify rewards
        q, score, adv = get_policy_loss(rewards_deltas, gamma=args.gamma,
                                        logits_seq=logits_seq, coo_actions=coo_actions)


        '''compute entropy loss'''
        m = Categorical(logits=logits_seq)
        loss_ent = m.entropy() # (bs, T)

        loss = (q - loss_ent * args.lamb_ent).sum()

        if phase == 'Train':
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(agent.parameters(), 10)
            optimizer_ag.step()

        '''calculate accuracy'''
        iou = torch.diagonal(box_iou(pred_boxes.float(),
                                     target), 0)
        correct += (iou >= 0.5).sum()

        pol_losses.append(q.sum().item())
        ent_losses.append(loss_ent.sum().item())
        rewards.append(torch.mean(rewards_deltas).item())

        if batch_idx % args.log_interval == 0:
            if batch_idx == 0:
                print('\n')
            print('{} Epoch: {} [{}/{} ({:.0f}%)]\tReward: {:.4f}'.format(
                phase, epoch, batch_idx * batch_size,
                len(data_loader.dataset),
                              100. * batch_idx / len_dataloader, np.mean(rewards)))

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

def evaluate_gen_acc(upload=True):
    agent_path = os.path.join(saveroot, 'best.pth.tar')
    '''load agent'''
    print('loading net ckpt from ', agent_path)
    ckpt = torch.load(agent_path)
    state_dict = ckpt['state_dict']
    epoch = ckpt['epoch']
    best_acc = ckpt['acc']
    agent.load_state_dict(state_dict)
    print(("=> loaded agent checkpoint epoch {} {}".format(epoch, best_acc)))

    kwargs = {'num_workers': 8, 'pin_memory': True}
    test_transform = Compose([Resize(IMG_SIZE)])
    test_accs = []
    with torch.no_grad():
        for d in range(0, 10):
            if d == args.digit:
                continue
            testset = MNIST_Corrupted(root='.', train=False, digit=d, bg_name=args.bg_name,
                                  datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist_github',
                                  transform=test_transform
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

if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ['WANDB_NAME'] = args.savename
    wandb.init(project="selfpaced")
    saveroot = os.path.join('/research/cbim/vast/tl601/results/loc-agent/',
                            args.savename)
    os.makedirs(saveroot, exist_ok=True)

    '''set dataloader'''
    if args.dataset == 'mnist':
        transform = Compose([Resize(args.img_size)])
    elif args.dataset == 'cub' or args.dataset == 'coco':
        anchors = generate_boxes(base_size=16, feat_height=14, feat_width=14,
                                 min_box_side=25, img_size=args.img_size,
                                 feat_stride=16, ratios=np.linspace(0.3, 3.5, num=15),
                                 scales=np.array(range(2, 13)))
        transform = t.Compose([
            t.ConvertFromPIL(),
            t.ToPercentCoords(),
            t.Resize(args.img_size),
            t.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            t.ToTensor()  # no change to (0, 1)
        ])
    train_loader, test_loader = init_dataloader()
    ''''''

    net, agent = init_model()
    wandb.watch(net, log_freq=10)
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
                                                   step_size=args.steps_ag, gamma=0.1)

    if args.evaluate == 1:
        epoch=0
        evaluate_gen_acc(False)
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
            if save_model:
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
    DEBUG = False
    evaluate_gen_acc()




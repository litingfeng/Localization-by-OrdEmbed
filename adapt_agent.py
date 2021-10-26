"""
Created on 10/26/21 10:50 AM
@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import os, wandb, torch
import numpy as np
from opts import parser
import torch.nn.functional as F
from itertools import product
from util import joint_transforms as t
from util.augmentations import Compose
from torch.distributions import Categorical
from util.data_aug import Resize
from datasets.clutter_mnist import MNIST_Corrupted
from datasets.cub import CUB
from datasets.coco_onecls import CocoDataset
from models.mnist_model import AutoencoderProj_ag
from models.agent_model import Agent
from models.cub_model import Encoder_CUB
from util.actions import Actor
from util.utils import box_iou
from RL import get_policy_loss

DEBUG = False

def init_dataloader():
    kwargs = {'num_workers': 8, 'pin_memory': True}
    if args.dataset == 'mnist':
        trainset = MNIST_Corrupted(root='.', train=True, digit=args.digit,
                               bg_name=args.bg_name, support_size=args.support_size,
                               sample_size=args.sample_size,
                               datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist_github',
                               transform=transform)
        testset = MNIST_Corrupted(root='.', train=False, digit=args.digit,
                              bg_name=args.bg_name, support_size=args.support_size,
                              datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist_github',
                              transform=transform)
    elif args.dataset == 'cub':
        assert (args.img_size == 224)
        trainset = CUB('/research/cbim/vast/tl601/Dataset/CUB_200_2011',
                        support_size=args.support_size, mode='base',
                       bg_name=args.bg_name, img_size=args.img_size, transform=transform)
        testset = CUB('/research/cbim/vast/tl601/Dataset/CUB_200_2011',
                        support_size=args.support_size, mode='val',
                      bg_name=args.bg_name, img_size=args.img_size, transform=transform)
    elif args.dataset == 'coco':
        assert (args.img_size == 224)
        trainset = CocoDataset(root='/research/cbim/vast/tl601/Dataset/coco/train2017',
                               support_size=args.support_size,
                               annFile='/research/cbim/vast/tl601/Dataset/coco/annotations/instances_train2017.json',
                               selected_cls=args.sel_cls, transforms=transform)
        testset = CocoDataset(root='/research/cbim/vast/tl601/Dataset/coco/val2017',
                              support_size=args.support_size,
                              annFile='/research/cbim/vast/tl601/Dataset/coco/annotations/instances_val2017.json',
                              selected_cls=args.sel_cls, transforms=transform)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    print('total train image: ', len(train_loader.dataset), ' test image: ',
          len(test_loader.dataset))
    return train_loader, test_loader

def init_model():
    if args.dataset == 'mnist':
        net = AutoencoderProj_ag(channel=1, pooling_size=args.pooling_size,
                                 dim=args.dim, pooling_mode=args.pooling_mode).to(device)
    elif args.dataset == 'cub' or args.dataset == 'coco':
        net = Encoder_CUB(pooling_size=args.pooling_size, pretrained=True, base=args.backbone,
                          dim=args.dim, pooling_mode=args.pooling_mode).to(device)
    agent = Agent(rnn=args.rnn, dim=args.dim_ag, poolsize=args.pooling_size,
                  num_class=args.num_act, hidden_size=args.hidden_size).to(device)
    '''load pretrained net'''
    if args.pretrained != '':
        ckpt = torch.load(os.path.join('/research/cbim/vast/tl601/results/loc-agent/', args.pretrained))
        mnist_pretrained_dict = ckpt['state_dict']
        embeddingnet_dict = net.state_dict()

        filtered_dict = {}
        for k, v in mnist_pretrained_dict.items():
            if k in embeddingnet_dict:
                #print('load ', k)
                filtered_dict[k] = v
        embeddingnet_dict.update(filtered_dict)
        net.load_state_dict(embeddingnet_dict)
        print('loaded from {}\nckpt epoch {} acc {:.2f}%'.format(args.pretrained,
                                                                 ckpt['epoch'], ckpt['acc_inst']))

    '''load pretrained agent'''
    if args.pretrained_agent != '':
        agent_path = os.path.join('/research/cbim/vast/tl601/results/loc-agent/',
                                  args.pretrained_agent)
        print('loading ckpt from ', agent_path)
        ckpt = torch.load(agent_path)
        state_dict = ckpt['state_dict']
        epoch = ckpt['epoch']
        best_loss = ckpt['acc']
        agent.load_state_dict(state_dict)
        print(("=> loaded checkpoint epoch {} {}".format(epoch, best_loss)))

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
    correct = 0
    len_dataloader = len(data_loader)
    for batch_idx, (data, target) in enumerate(data_loader):
        batch_size = data.shape[0]
        data, target = data.to(device),  target.float()
        optimizer_ag.zero_grad()

        '''initialize'''
        actor = Actor(data.shape[2:], data.shape[0], min_box_side=args.min_box_side)
        rewards_all = torch.zeros((batch_size, args.seq_len+1)).to(device)
        action_seq = torch.IntTensor(batch_size, args.seq_len).to(device)
        logits_seq, boxes, all_embed = [], [], []
        if args.hidden_size:
            h_t = torch.zeros(
                batch_size,
                args.hidden_size,
                dtype=torch.float,
                device=device,
                requires_grad=True,
            )
        # start from the whole image
        pred_boxes = torch.tensor([[0, 0, args.img_size-1, args.img_size-1]]).float().repeat(batch_size, 1).to(device)
        rois = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                          pred_boxes), dim=1)

        '''forward embedding net'''
        with torch.no_grad():
            embed, top_feat = model(data, rois)

        '''compute reward of the first step '''
        rewards_all[:, 0] = F.pairwise_distance(proto_feat, top_feat)

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
            all_embed.append(top_feat)

            action_seq[:, t] = actions.view(1, -1)
            logits_seq.append(logits)
            boxes.append(pred_boxes)

        rewards_trns = F.pairwise_distance(proto_feat, torch.cat(all_embed))
        rewards_all[:, 1:] = rewards_trns.view(-1, batch_size).permute(1, 0)

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
            optimizer_ag.step()

        '''calculate accuracy'''
        iou = torch.diagonal(box_iou(pred_boxes.float(),
                                     target[:, :4]), 0)
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

def evaluate_acc(upload=True):
    agent_path = os.path.join(saveroot, 'best.pth.tar')
    '''load agent'''
    print('loading net ckpt from ', agent_path)
    ckpt = torch.load(agent_path)
    state_dict = ckpt['state_dict']
    epoch = ckpt['epoch']
    best_acc = ckpt['acc']
    agent.load_state_dict(state_dict)
    print(("=> loaded agent checkpoint epoch {} {}".format(epoch, best_acc)))
    if upload:
        wandb.run.summary.update({'Test Best Loc Acc': best_acc})

    test_loss, test_acc = feed_data(net, agent, data_loader=train_loader, is_train=False, upload=False)
    print('Train Loc Acc ', test_acc)
    wandb.run.summary.update({'Train Best Loc Acc': test_acc})

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
        transform = t.Compose([
            t.ConvertFromPIL(),
            t.ToPercentCoords(),
            t.Resize(args.img_size),
            t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            t.ToTensor()  # no change to (0, 1)
        ])
    train_loader, test_loader = init_dataloader()
    ''''''

    net, agent = init_model()
    wandb.watch(net, log_freq=10)
    wandb.watch(agent, log_freq=10)
    wandb.config.update(args)

    '''compute prototype embedding'''
    supp_data, supp_target = train_loader.dataset.supp_data.to(device), \
                             train_loader.dataset.supp_targets.float()
    rois_supp = torch.cat((torch.arange(0, supp_data.shape[0]).float().view(-1, 1),
                         supp_target), dim=1).to(device)
    with torch.no_grad():
        supp_embed, supp_top_feat = net(supp_data, rois_supp)  # (support_size, 512)
    proto_feat = torch.mean(supp_top_feat, dim=0)  # (512,)
    ''''''

    if args.optimizer_ag == 'SGD':
        optimizer_ag = torch.optim.SGD(agent.parameters(), lr=args.lr_ag, momentum=0.9)
    elif args.optimizer_ag == 'Adam':
        optimizer_ag = torch.optim.Adam(agent.parameters(), lr=args.lr_ag)
    scheduler_ag = torch.optim.lr_scheduler.StepLR(optimizer_ag, step_size=args.steps_ag, gamma=0.1)

    # evaluate on new digits before finetuning
    epoch = 0
    test_loss, test_acc = feed_data(net, agent, data_loader=test_loader, is_train=False, upload=False)
    test_loss, train_acc = feed_data(net, agent, data_loader=train_loader, is_train=False, upload=False)
    print('before finetuning train_acc: {:.2f} test_acc: {:.2f}'.format(train_acc, test_acc))
    wandb.run.summary.update({'Train Before Loc Acc': train_acc,
                              'Test Before Loc Acc': test_acc})

    if args.evaluate == 1:
        epoch=0
        evaluate_acc(True)
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
    evaluate_acc()






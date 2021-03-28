"""
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
import pickle
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from actions import Actor
from util import cosine_simililarity
from torchvision import datasets, transforms
from RL import get_policy_loss

DEBUG = False

def sample(logits):
    return torch.multinomial(logits, 1)

def selector(prob, pred, target, sel_conf=True):
    conf = (prob.flatten() >= thresh).nonzero().flatten()
    correct_inds = (pred.eq(target.view_as(pred))).squeeze().nonzero().flatten()
    combined = torch.cat((conf, correct_inds))
    uniques, counts = combined.unique(return_counts=True)
    confidents = uniques[counts > 1]
    if not sel_conf:
        combined = torch.cat((confidents, torch.arange(prob.shape[0]).to(device)))
        uniques, counts = combined.unique(return_counts=True)
        less_conf = uniques[counts == 1]
        return confidents, less_conf
    return confidents

def get_proto():
    correct = 0
    with torch.no_grad():
        p0, p1 = [], []
        for batch_idx, (data, target, _) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            feature, output = net(data)
            output = F.softmax(output, dim=1)
            print('output ', output.shape)
            prob, pred = output.max(1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            #print('rpob ', prob.shape, ' pred ', pred.shape)
            confidents = selector(prob, pred, target)
            conf_feature, conf_pred = feature[confidents], pred[confidents].flatten()
            mask_target_0 = (conf_pred == 0)
            conf_feature_pro0, conf_feature_pro1 = conf_feature[mask_target_0 == True], \
                                                   conf_feature[mask_target_0 == False]
            p0.append(conf_feature_pro0)
            p1.append(conf_feature_pro1)
        p0 = torch.cat(p0)
        p1 = torch.cat(p1)
        proto = torch.stack((p0.mean(0), p1.mean(0)))

    acc = 100. * correct / len(test_loader.dataset)
    print('acc in eval mode: {:.2f}%'.format(acc))
    exit()

    return proto

def evaluate(net, dataloader):
    net.eval()
    correct = 0
    embed, targets, probs = [], [], []
    with torch.no_grad():
        for batch_idx, (data, target, _) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            feature, output = net(data)
            output = F.softmax(output, dim=1)
            prob, pred = output.max(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            embed.append(feature)
            targets.append(target)
            probs.append(prob)
    acc = 100. * correct / len(dataloader.dataset)
    print('in eval mode: acc {:.2f}%'.format(acc))

    return acc

# dataloader class
class myCIFAR(datasets.CIFAR10):
    def __init__(self, root, subset, ratio=0.3, datapath=None, train=True, transform=None,
                 target_transform=None, download=True):
        super(myCIFAR, self).__init__(root, train=train, download=download,
                                      transform=transform, target_transform=target_transform)
        self.subset = subset
        self.ratio = ratio
        phase = 'train' if train else 'test'
        if not datapath:
            # select 4, 1
            self.newdata, self.newtargets = [], []
            for idx, digit in enumerate(self.targets):
                if digit == 4 or digit == 1:
                    self.newdata.append(idx)
                    target = 0 if digit == 4 else 1
                    self.newtargets.append(target)
            self.newdata, self.newtargets = np.array(self.newdata), \
                                        np.array(self.newtargets)
            self.rot_inds = np.random.choice(len(self.newdata),
                                             size=int(self.ratio*len(self.newdata)), replace=False) # select 100 to rotate
            print('rot number of 4 {}\nrot number of 1 {}'.format(
                len(np.where(self.newtargets[self.rot_inds] == 0)[0]),
                len(np.where(self.newtargets[self.rot_inds] == 1)[0])))
            self.norot_inds =np.array(list(set(range(len(self.newdata))) - set(self.rot_inds)))
            print('self.norot_inds ', self.norot_inds.shape)
            pickle.dump((self.newdata, self.newtargets, self.rot_inds, self.norot_inds),
                        open('{}_cifar_rot{}_more.pkl'.format(phase,
                                                              int(self.ratio*100)), 'wb'))
        else:
            self.newdata, self.newtargets, self.rot_inds, self.norot_inds = \
                pickle.load(open('{}_cifar_rot{}_more.pkl'.format(phase,
                                                                  int(self.ratio*100)), 'rb'))
        print('number of 4 {}\nnumber of 1 {}'.format(len(np.where(self.newtargets == 0)[0]),
                                                      len(np.where(self.newtargets == 1)[0])))
        print(' rot: {} norot_inds: {} '.format(self.rot_inds.shape, self.norot_inds.shape))

        # select which subset to train
        if self.subset == 'original_all': # use all original(non-roated) 200 samples
            self.data = [self.data[i] for i in self.newdata]
            self.targets = self.newtargets
            self.data = np.stack(self.data)
        elif self.subset == 'original_half': # use 1-ratio non-rotated samples
            print('self.norot_inds ', self.norot_inds.shape)
            self.data = [self.data[self.newdata[i]] for i in self.norot_inds]
            self.targets = self.newtargets[self.norot_inds]
            self.data = np.stack(self.data)
        elif self.subset == 'half_half':
            num_each = int(len(self.rot_inds) / 3.)
            self.orgdata = [self.data[i].transpose(2,0,1) for i in self.newdata] # HWC->CHW
            self.data = copy.deepcopy(self.orgdata)
            for i, inds in enumerate(self.rot_inds):
                k = i // num_each + 1
                self.data[inds] = np.rot90(
                    self.data[inds].copy(), k, axes=(-2,-1))
            self.data = np.stack(self.data).transpose(0, 2, 3, 1)
            self.targets = self.newtargets
            self.rot_dict = torch.zeros(len(self.data))
            self.rot_dict[self.rot_inds] = 1
        else:
            print('Not implementd')
            exit()

        print('subset [{}] data: {}'.format(self.subset, self.data.shape[0]))

        if self.subset == 'half_half':
            self.four_rot, self.one_rot = [], []
            for i in self.rot_inds:
                self.four_rot.append(i) if self.targets[i] == 0 else self.one_rot.append(i)
            self.four_norot, self.one_norot = [], []
            for i in self.norot_inds:
                self.four_norot.append(i) if self.targets[i] == 0 else self.one_norot.append(i)
            print('rot 4: {} rot 1: {}'.format(len(self.four_rot), len(self.one_rot)))
            print('nonrot 4: {} nonrot 1: {}'.format(len(self.four_norot), len(self.one_norot)))

    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]
        img = Image.fromarray(data)

        if DEBUG:
            for i in (10, 1200, 2200):
                inds = self.rot_inds[i]
                data, target = self.data[inds], self.targets[inds]
                img = Image.fromarray(data)
                f, ax = plt.subplots()
                ax.imshow(np.asarray(
                    Image.fromarray(self.orgdata[inds].transpose(1, 2, 0))), cmap='gray', interpolation='none')
                plt.title('org')
                print('target ', target)
                plt.show()

                f, ax = plt.subplots()
                ax.imshow(np.asarray(img), cmap='gray', interpolation='none')
                plt.title('trans')
                plt.show()
            exit()
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(1600, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 5)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class Net(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(Net, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(5, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        feature = self.nonlinear(output)
        scores = self.fc1(feature)
        return feature, scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))

class Agent(nn.Module):
    def __init__(self, dim=2, num_action=2):
        super(Agent, self).__init__()
        self.num_action = num_action
        self.fc = nn.Sequential(
            nn.Linear(dim, 8),
            nn.ReLU(),
            nn.Linear(8, self.num_action)
        )
        self.bn1 = nn.BatchNorm1d(2)

    def forward(self, feature):
        #self.bn1 = nn.BatchNorm1d(2)
        logits = self.fc(feature)
        draw = sample(F.softmax(logits, dim=1))
        return logits, draw

if __name__ == "__main__":
    # seed = 14
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # np.random.seed(seed)  # Numpy module.
    # random.seed(seed)  # Python random module.
    #torch.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    #random_seed = 2

    # torch.backends.cudnn.enabled = False
    # torch.manual_seed(random_seed)

    #os.environ['WANDB_NAME'] = "binary_classification"
    #wandb.init(project="selfpaced")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = myCIFAR(root='.', subset='half_half', ratio=0.5,
                      datapath='.',
                      train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]))

    test_set = myCIFAR(root='.', subset='original_all', ratio=0.5,
                       datapath='.',
                       train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]))
    train_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=512, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_set,
                                            batch_size=512, shuffle=False, num_workers=8)
    print('total images of 4 & 1: ', len(dataset), ' test: ', len(test_set))

    run_accs = {'train': [], 'test': [], 'train_r': []}
    for run in range(10):
        thresh = 0.001
        embedding_net = EmbeddingNet()
        net = Net(embedding_net, 2).to(device)
        agent = Agent(dim=5, num_action=2).to(device)
        #load pretrained classifier
        #ckpt = torch.load('half_half_large.pth.tar')
        #ckpt = torch.load('original_half_rot30_large.pth.tar')
        ckpt = torch.load('/research/cbim/vast/tl601/results/'
                          'selfpaced/ordinal/ordinal_loss_triplet_st80_m1_lm1_original_half/'
                          'best.pth.tar')
        net.load_state_dict(ckpt['state_dict'])
        print('loaded ckpt epoch {} acc {:.2f}%'.format(ckpt['epoch'], ckpt['acc']))

        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        optimizer_agent = torch.optim.Adam(agent.parameters(), lr=0.001)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.1)
        criterion2 = nn.CrossEntropyLoss(reduction='none')

        '''
        training
        '''
        net.train()
        agent.train()
        total_loss, total_pol_losses, total_rewards = [], [], []
        total_org_loss = []
        total_r_back = []
        accs, accs_r_back = [], []
        less_confs, new_less_confs = [], []
        epochs = 15
        actor = Actor()
        for epoch in range(epochs):
            # update proto
            # proto = get_proto()
            # print('proto ', proto.shape)
            train_losses, pol_losses, rewards = [], [], []
            train_org_losses = []
            r_back = []
            correct, correct_r_back = 0, 0
            for batch_idx, (data, target, index) in enumerate(train_loader):
                batch_size = data.shape[0]
                data, target = data.to(device), target.to(device)
                isrot = dataset.rot_dict[index].to(device)
                conf = (isrot == 0).nonzero().flatten()
                #
                optimizer.zero_grad()
                optimizer_agent.zero_grad()

                # forward classifier
                with torch.no_grad():
                    feature, output = net(data)
                    loss_org = criterion(output, target)
                    # loss[less_conf] = 0
                    # loss = loss.mean()
                    output = F.softmax(output, dim=1)
                    prob, pred = output.max(1, keepdim=True)

                # select less confident data and take actions
                less_conf = (loss_org.flatten() > thresh).nonzero().flatten()
                #conf, less_conf = selector(prob, pred, target, sel_conf=False)
                #less_conf = torch.arange(0, batch_size)
                #less_conf = (isrot == 1).nonzero().flatten()
                #less_conf = (prob.flatten() < 0.9).nonzero().flatten()
                # conf = (prob.flatten() >= thresh).nonzero().flatten()
                if len(less_conf) == 0:
                    print('no less confidents')
                    continue

                logits, actions = agent(feature[less_conf].detach())
                #print('act ', actions.shape)

                #actions = torch.randint(0,2, size=actions.shape)
                # actions = torch.from_numpy(np.random.choice(2,
                #                     size=(len(less_conf),), p=[0.8,0.2])).view_as(actions)
                data[less_conf] = actor.takeaction(data[less_conf], actions)
                correct_r_back += torch.logical_and(actions.squeeze() == 0,
                                                    isrot[less_conf] == 1).sum().item()
                #correct_r_back += (actions.squeeze() == 0).sum().item()
                #r_back.append(100. * (actions == 0).sum().item() / len(less_conf))

                # compute reward
                tran_feature, tran_output = net(data)
                loss = criterion(tran_output, target).mean()
                loss.backward()
                optimizer.step()
                tran_output = F.softmax(tran_output, dim=1)
                tran_prob, tran_pred = tran_output.max(1, keepdim=True)

                #new_less_conf = len((tran_prob.flatten() < thresh).nonzero().flatten())
                keep = 10 if len(less_conf) > 10 else len(less_conf)
                if keep == 0:
                    print(' less conf 0')
                print('actions ', actions[:keep].view(-1))
                # print('rotate back  {:.2f}%'.format(
                #     100. * (actions==0).sum().item() / len(less_conf)) )
                print('is rotated    ', dataset.rot_dict[index[less_conf[:keep]]])
                print('target ', target[less_conf[:keep]])
                print('pred   ', pred[less_conf[:keep]].view(-1))
                print('t_pred ', tran_pred[less_conf[:keep]].view(-1))
                print('prob   ', prob[less_conf[:keep]].view(-1))
                print('tran_prob ', tran_prob[less_conf[:keep]].view(-1))

                #if actions[:0] == 1 and pred[less_conf[0]][0] == tran_pred[0]

                new_less_conf = len(selector(tran_prob[less_conf],
                                             tran_pred[less_conf], target[less_conf]))
                correct += tran_pred.eq(target.view_as(tran_pred)).sum().item()

                # similarity_matrix = cosine_simililarity(tran_feature, proto)
                # similarity_matrix_org = cosine_simililarity(feature[less_conf], proto)
                # reward = (- criterion2(similarity_matrix, target[less_conf])) - \
                # (-criterion2(similarity_matrix_org, target[less_conf]))

                # only consider prob for target class
                trn_prob = torch.gather(tran_output[less_conf], 1, target[less_conf].unsqueeze(1))
                org_prob = torch.gather(output[less_conf], 1, target[less_conf].unsqueeze(1))
                reward = (trn_prob - org_prob)
                #reward = torch.sign(reward)

                # if correct and prob goes up, reward; if wrong, -1
                # corrects = (tran_pred[less_conf].view_as(target[less_conf]) == target[less_conf])
                # incorrects = (tran_pred[less_conf].view_as(target[less_conf]) != target[less_conf])
                # reward = (tran_prob[less_conf] - prob[less_conf]).flatten() * corrects \
                #          + (-1)*incorrects
                #reward = - criterion2(similarity_matrix, target[less_conf])
                #reward = - criterion2(similarity_matrix, tran_pred.view(-1))
                # print('org_prob ', output[less_conf][:keep])
                # print('trn_prob ', tran_output[less_conf][:keep])
                # print('reward ', reward[:keep].view(-1))

                coord = product(range(actions.size(0)), range(actions.size(1)))
                coo_actions = [[k, m, actions[k, m]] for k, m in coord]
                q, score, critic_loss, adv = get_policy_loss(reward.view(-1, 1), len(less_conf), gamma=0.99,
                                                             logits_seq=logits.view(len(less_conf), 1, -1),
                                                             seq_len=1,
                                                             coo_actions=coo_actions, values=None)

                q.backward()
                optimizer_agent.step()
                pol_losses.append(q.item())
                rewards.append(torch.mean(reward).item())
                train_losses.append(loss.item())
                train_org_losses.append(loss_org.mean().item())
                less_confs.append(len(less_conf))
                new_less_confs.append(new_less_conf)
                r_back.append(100. * torch.logical_and(actions.squeeze() == 0,
                                                    isrot[less_conf] == 1).sum().item()
                              / (len((isrot[less_conf] == 1).nonzero().flatten())+1e-6))
                print('coor_r_back ', torch.logical_and(actions.squeeze() == 0,
                                                    isrot[less_conf] == 1).sum().item())
                print('r_bakc ', 100. * torch.logical_and(actions.squeeze() == 0,
                                                    isrot[less_conf] == 1).sum().item()
                              / (len((isrot[less_conf] == 1).nonzero().flatten())+1e-6),
                      '\n less_conf ', len((isrot[less_conf] == 1).nonzero().flatten()))
                print('r_back in batch ', 100. * len((actions.squeeze() == 0).nonzero().flatten()) /
                      batch_size   )
                # r_back.append(100. * (actions.squeeze() == 0).sum().item()
                #               / (len(less_conf) + 1e-6))

            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tCls Loss: {:.4f}\tR: {:.4f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #            100. * batch_idx / len(train_loader),
            #            np.mean(train_losses), np.mean(rewards)))
            #thresh = thresh * 1.3
            total_loss.append(np.mean(train_losses))
            total_org_loss.append(np.mean(train_org_losses))
            total_pol_losses.append(np.mean(pol_losses))
            total_rewards.append(np.mean(rewards))
            total_r_back.append(np.mean(r_back))
            acc = 100. * correct / len(train_loader.dataset)
            acc_r_back = 100. * correct_r_back / len(dataset.rot_inds)
            accs.append(acc)
            accs_r_back.append(acc_r_back)
            torch.save({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'acc': acc
            }, 'rotate_rl_joint_all.pth.tar')
            #break
        # #plot loss and acc
        # fig = plt.figure()
        # plt.plot(range(len(less_confs)), less_confs, label='less_conf')
        # plt.plot(range(len(new_less_confs)), new_less_confs, label='new_less_conf')
        # plt.legend()
        # plt.xlabel('iteration')
        # plt.ylabel('less_confs')
        # plt.show()
        # fig = plt.figure()
        # plt.plot(range(epochs), total_loss, label='train loss')
        # plt.plot(range(epochs), total_org_loss, label='train_org_loss')
        # plt.legend()
        # plt.xlabel('epoch')
        # plt.ylabel('loss')
        # plt.show()
        # fig = plt.figure()
        # plt.plot(range(len(total_pol_losses)), total_pol_losses)
        # plt.legend(['Train Pol Loss'], loc='upper right')
        # plt.xlabel('epoch')
        # plt.ylabel('loss')
        # plt.show()
        fig = plt.figure()
        plt.plot(range(len(total_r_back)), total_r_back)
        plt.legend(['% of rotate back'], loc='upper right')
        plt.xlabel('epoch')
        plt.ylabel('percent of rotate back')
        plt.show()
        # fig = plt.figure()
        # plt.plot(range(len(total_rewards)), total_rewards)
        # plt.legend(['Train Rewards'], loc='upper right')
        # plt.xlabel('epoch')
        # plt.ylabel('reward')
        # plt.show()
        fig = plt.figure()
        plt.plot(range(epochs), accs, label='acc')
        plt.plot(range(epochs), accs_r_back, label='acc_r_back')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('accuracy (%)')
        plt.show()
        run_accs['train'].append(acc)
        run_accs['train_r'].append(acc_r_back)
        '''
        testing
        '''
        ckpt = torch.load('rotate_rl_joint_all.pth.tar')
        net.load_state_dict(ckpt['state_dict'])
        print('loaded ckpt epoch {} acc {:.2f}%'.format(ckpt['epoch'], ckpt['acc']))
        #evaluate(net, agent, train_loader_noshuf)
        acc = evaluate(net, test_loader)
        run_accs['test'].append(acc)
        print('acc_r_back ', acc_r_back)
        exit()

    print('run accs\ntrain\n', run_accs['train'])
    print('\ntest\n', run_accs['test'])
    print('train acc: {:.2f}+-{:.2f}\n'
          'test acc: {:.2f}+-{:.2f}'.format(
        np.mean(run_accs['train']),
        np.std(run_accs['train']),
        np.mean(run_accs['test']),
        np.std(run_accs['test'])))

    print('run accs_r ', run_accs['train_r'])
    print('acc_r_back: {:.2f}+-{:.2f}'.format(np.mean(run_accs['train_r']),
        np.std(run_accs['train_r'])))




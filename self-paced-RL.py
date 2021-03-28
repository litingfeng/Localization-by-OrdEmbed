"""
finetune classifier with RL, self-paced
Created on 2/18/2021 7:18 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import os, wandb
#import IPython.embed as emb
import torch
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
def plot_embedding(embed, targets, probs, acc, trans=False):
    embed, probs = torch.cat(embed, dim=0), \
                            torch.cat(probs)
    embed, targets, probs = embed.cpu().numpy(), targets.cpu().numpy(), \
                            probs.cpu().numpy().flatten()
    less_confidents = np.argwhere(probs <= thresh).flatten()
    print('percent of less confidents {:.2f}%'.format(100. * len(less_confidents) / len(probs)))

    # embeddings
    seven, one = np.argwhere(targets == 0).flatten(), np.argwhere(targets == 1).flatten()
    print('seven ', seven.shape, ' one ', one.shape)
    print('seven_rot ', len(dataset.seven_rot), ' one_rot ', len(dataset.one_rot))
    print('seven_norot ', len(dataset.seven_norot), ' one_norot ', len(dataset.one_norot))

    fig = plt.figure()
    plt.scatter(embed[seven, 0], embed[seven, 1], color='g', label='7', s=2)
    plt.scatter(embed[dataset.seven_rot, 0], embed[dataset.seven_rot, 1], color='b', label='7_rot', s=2, alpha=0.5)
    plt.scatter(embed[one, 0], embed[one, 1], color='r', label='1', s=2, alpha=0.5)
    plt.scatter(embed[dataset.one_rot, 0], embed[dataset.one_rot, 1], color='y', label='1_rot', s=2, alpha=0.5)
    plt.scatter(embed[less_confidents, 0], embed[less_confidents, 1], color='c', label='less_conf', s=2, alpha=0.5)
    plt.legend()
    if not trans:
        plt.title('digit 7 1 Rotate 30% Epoch {} Acc {:.2f}'.format(epochs, acc))
        plt.savefig('./images/Rotate_embed.png')
    else:
        plt.title('Trans digit 7 1 Rotate 30% Epoch {} Acc {:.2f}'.format(epochs, acc))
        plt.savefig('./images/Rotate_trans_embed.png')
    plt.show()

def get_proto():
    with torch.no_grad():
        p0, p1 = [], []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            feature, output = net(data)
            output = F.softmax(output, dim=1)
            prob, pred = output.max(1, keepdim=True)
            conf = (prob.flatten() >= thresh).nonzero().flatten()
            conf_feature, conf_pred = feature[conf], pred[conf].flatten()
            mask_target_0 = (conf_pred == 0)
            conf_feature_pro0, conf_feature_pro1 = conf_feature[mask_target_0 == True], \
                                                   conf_feature[mask_target_0 == False]
            p0.append(conf_feature_pro0)
            p1.append(conf_feature_pro1)
        p0 = torch.cat(p0)
        p1 = torch.cat(p1)
        proto = torch.stack((p0.mean(0), p1.mean(0)))

    return proto

# dataloader class
class myMNIST(datasets.MNIST):
    def __init__(self, root, datapath=None, train=True, transform=None, target_transform=None,
                 download=False):
        super(myMNIST, self).__init__(root, train=train, download=download,
                                      transform=transform, target_transform=target_transform)
        if not datapath:
            # select 7, 1
            self.newdata, self.newtargets = [], []
            for idx, digit in enumerate(self.targets):
                if digit == 7 or digit == 1:
                    self.newdata.append(idx)
                    target = 0 if digit == 7 else 1
                    self.newtargets.append(target)

            # randomly sample 30% to rotate 90 degree
            inds = np.random.permutation(len(self.newdata))
            self.rot_inds = inds[:int(0.3*len(self.newdata))]
            self.norot_inds = inds[int(0.3*len(self.newdata)):]
            pickle.dump((self.newdata, self.newtargets, self.rot_inds, self.norot_inds),
                        open('train_rotate.pkl', 'wb'))
        else:
            self.newdata, self.newtargets, self.rot_inds, self.norot_inds = \
                pickle.load(open('train_rotate.pkl', 'rb'))
        self.seven_rot, self.one_rot = [], []
        for i in self.rot_inds:
            data = self.data[self.newdata[i]].numpy()
            # counter clockwise
            self.data[self.newdata[i]] = torch.from_numpy(np.rot90(data).copy())
            self.seven_rot.append(i) if self.newtargets[i] == 0 else self.one_rot.append(i)
        self.seven_norot, self.one_norot = [], []
        for i in self.norot_inds:
            self.seven_norot.append(i) if self.newtargets[i] == 0 else self.one_norot.append(i)

    def __getitem__(self, index):
        #index = self.rot_inds[10]
        data, target = self.data[self.newdata[index]], self.newtargets[index]
        img = Image.fromarray(data.numpy(), mode='L')

        if DEBUG:
            f, ax = plt.subplots()
            ax.imshow(np.asarray(img), cmap='gray', interpolation='none')
            print('target ', target)
            plt.show()
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.newdata)

    @property
    def raw_folder(self):
        return os.path.join('/research/cbim/vast/tl601/Dataset', 'MNIST', 'raw')

    @property
    def processed_folder(self):
        return os.path.join('/research/cbim/vast/tl601/Dataset', 'MNIST', 'processed')

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
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
        self.fc1 = nn.Linear(2, n_classes)

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

    def forward(self, feature):
        logits = self.fc(feature)
        draw = sample(F.softmax(logits, dim=1))
        return logits, draw

if __name__ == "__main__":
    random_seed = 1
    thresh = 0.9
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    #os.environ['WANDB_NAME'] = "binary_classification"
    #wandb.init(project="selfpaced")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = myMNIST(root = '.', datapath='.',
                      train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    train_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=512, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=512, shuffle=False, num_workers=8)
    print('total images of 7 & 1: ', len(dataset))

    embedding_net = EmbeddingNet()
    net = Net(embedding_net, 2).to(device)
    agent = Agent(dim=2, num_action=2).to(device)
    # load pretrained classifier
    ckpt = torch.load('rotate.pth.tar')
    net.load_state_dict(ckpt['state_dict'])
    print('loaded ckpt epoch {} acc {:.2f}%'.format(ckpt['epoch'], ckpt['acc']))

    criterion = nn.CrossEntropyLoss()
    #wandb.watch(net)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.5)
    optimizer_agent = torch.optim.Adam(agent.parameters(), lr=0.001)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.1)
    criterion2 = nn.CrossEntropyLoss(reduction='none')

    # intialize proto
    proto = get_proto()

    '''
    training
    '''
    net.train()
    total_loss, total_pol_losses, total_rewards = [], [], []
    accs, accs_new = [], []
    less_confs, new_less_confs = [], []
    epochs = 10
    actor = Actor()
    # for epoch in range(epochs):
    #     train_losses, pol_losses, rewards = [], [], []
    #     correct, correct_new = 0, 0
    #     for batch_idx, (data, target) in enumerate(train_loader):
    #         batch_size = data.shape[0]
    #         data, target = data.to(device), target.to(device)
    #         #optimizer.zero_grad()
    #         optimizer_agent.zero_grad()
    #
    #         #print('data ', data.shape, 'target ', target.shape)
    #         # forward classifier
    #         feature, output = net(data)
    #         output = F.softmax(output, dim=1)
    #         prob, pred = output.max(1, keepdim=True)
    #         loss = criterion(output, target)
    #         #loss.backward()
    #         #optimizer.step()
    #         train_losses.append(loss.item())
    #
    #         # select less confident data and take actions
    #         less_conf = (prob.flatten() < thresh).nonzero().flatten()
    #         conf = (prob.flatten() >= thresh).nonzero().flatten()
    #         if len(less_conf) == 0: continue;
    #         logits, actions = agent(feature[less_conf].detach())
    #         data[less_conf] = actor.takeaction(data[less_conf], actions)
    #         correct += pred.eq(target.view_as(pred)).sum().item()
    #
    #         # compute reward
    #         with torch.no_grad():
    #             tran_feature, tran_output = net(data[less_conf])
    #             tran_prob, tran_pred = F.softmax(tran_output,
    #                                              dim=1).max(1, keepdim=True)
    #             new_less_conf = len((tran_prob.flatten() < thresh).nonzero().flatten())
    #             # conf_feature, conf_pred = feature[conf], pred[conf].flatten()
    #             # mask_target_0 = (conf_pred == 0)
    #             # conf_feature_pro0, conf_feature_pro1 = conf_feature[mask_target_0==True], \
    #             #                                      conf_feature[mask_target_0==False]
    #             pred[less_conf] = tran_pred
    #             correct_new += pred.eq(target.view_as(pred)).sum().item()
    #             # conf_feature_pro0, conf_feature_pro1 = conf_feature_pro0.mean(dim=0), \
    #             #                                        conf_feature_pro1.mean(dim=0)
    #             # proto = torch.stack((conf_feature_pro0, conf_feature_pro1), dim=0)
    #             similarity_matrix = cosine_simililarity(tran_feature, proto)
    #             reward = - criterion2(similarity_matrix, tran_pred.view(-1))
    #
    #         coord = product(range(actions.size(0)), range(actions.size(1)))
    #         coo_actions = [[k, m, actions[k, m]] for k, m in coord]
    #         q, score, critic_loss, adv = get_policy_loss(reward.view(-1, 1), len(less_conf), gamma=0.99,
    #                                                      logits_seq=logits.view(len(less_conf), 1, -1),
    #                                                      seq_len=1,
    #                                                      coo_actions=coo_actions, values=None)
    #
    #         q.backward()
    #         optimizer_agent.step()
    #         pol_losses.append(q.item())
    #         rewards.append(torch.mean(reward).item())
    #         less_confs.append(len(less_conf))
    #         new_less_confs.append(new_less_conf)
    #
    #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tCls Loss: {:.4f}\tR: {:.4f}'.format(
    #         epoch, batch_idx * len(data), len(train_loader.dataset),
    #                100. * batch_idx / len(train_loader),
    #                np.mean(train_losses), np.mean(rewards)))
    #
    #     total_loss.append(np.mean(train_losses))
    #     total_pol_losses.append(np.mean(pol_losses))
    #     total_rewards.append(np.mean(rewards))
    #     acc = 100. * correct / len(train_loader.dataset)
    #     acc_new = 100. * correct_new / len(train_loader.dataset)
    #     accs.append(acc)
    #     accs_new.append(acc_new)
    #     torch.save({
    #         'epoch': epoch,
    #         'state_dict': net.state_dict(),
    #         'acc': acc_new
    #     }, 'rotate_rl.pth.tar')
    #     #break
    # #plot loss and acc
    # fig = plt.figure()
    # plt.plot(range(len(less_confs)), less_confs, label='less_conf')
    # plt.plot(range(len(new_less_confs)), new_less_confs, label='new_less_conf')
    # plt.legend()
    # plt.xlabel('iteration')
    # plt.ylabel('less_confs')
    # plt.show()
    # fig = plt.figure()
    # plt.plot(range(epochs), total_loss)
    # plt.legend(['Train Loss'], loc='upper right')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.show()
    # fig = plt.figure()
    # plt.plot(range(len(total_pol_losses)), total_pol_losses)
    # plt.legend(['Train Pol Loss'], loc='upper right')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.show()
    # fig = plt.figure()
    # plt.plot(range(len(total_rewards)), total_rewards)
    # plt.legend(['Train Rewards'], loc='upper right')
    # plt.xlabel('epoch')
    # plt.ylabel('reward')
    # plt.show()
    # fig = plt.figure()
    # plt.plot(range(epochs), accs, label='acc')
    # plt.plot(range(epochs), accs_new, label='acc_new')
    # plt.legend()
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy (%)')
    # plt.show()

    '''
    testing
    '''
    ckpt = torch.load('rotate_rl.pth.tar')
    net.load_state_dict(ckpt['state_dict'])
    print('loaded ckpt epoch {} acc {:.2f}%'.format(ckpt['epoch'], ckpt['acc']))
    # plot embedding
    net.eval()
    correct, correct_trans = 0, 0
    embed, targets, probs = [], [], []
    embed_new, probs_new = [], []
    preds, preds_new = [], []
    all_rot_inds = []
    num_instance = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            feature, output = net(data)
            output = F.softmax(output, dim=1)
            prob, pred = output.max(1, keepdim=True)
            # select less confidents and transform
            less_conf = (prob.flatten() < thresh).nonzero().flatten()
            logits, actions = agent(feature[less_conf].detach())
            rot_inds = (actions.squeeze() == 0).nonzero().flatten()
            all_rot_inds += [num_instance + i.item() for i in rot_inds]
            if batch_idx<2:
                print('batch ', batch_idx, '\n', all_rot_inds)
            num_instance += data.shape[0]
            data[less_conf] = actor.takeaction(data[less_conf], actions)
            feature_new, output = net(data)
            output = F.softmax(output, dim=1)
            prob_new, pred_new = output.max(1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()
            correct_trans += pred_new.eq(target.view_as(pred_new)).sum().item()

            embed.append(feature)
            embed_new.append(feature_new)
            targets.append(target)
            probs.append(prob)
            probs_new.append(prob_new)
            preds.append(pred)
            preds_new.append(pred_new)
    print('all_rot_inds ', len(all_rot_inds), all_rot_inds[:10])
    correct_trns = len(list(set(all_rot_inds).intersection(dataset.seven_rot+dataset.one_rot)))
    print('correct transfer back: ', correct_trns, '{:.2f}%'.format(
        100. * correct_trns / len(dataset.seven_rot+dataset.one_rot)))
    wrong_trns = len(list(set(all_rot_inds).intersection(dataset.seven_norot + dataset.one_norot)))
    print('wrong transfer back: ', wrong_trns, '{:.2f}%'.format(
        100. * wrong_trns / len(dataset.seven_norot + dataset.one_norot)))
    print('7 rot: ', len(dataset.seven_rot), ' 1 rot: ', len(dataset.one_rot))
    print('7 norot: ', len(dataset.seven_norot), ' 1 norot: ', len(dataset.one_norot))
    exit()
    acc = 100. * correct / len(test_loader.dataset)
    acc_new = 100. * correct_trans / len(test_loader.dataset)
    print('in eval mode: acc {:.2f}%\tacc_new {:.2f}'.format(acc, acc_new))
    preds, preds_new, targets = torch.cat(preds).squeeze(), \
                                torch.cat(preds_new).squeeze(), \
                                torch.cat(targets)
    acc_7_rot = 100. * preds[dataset.seven_rot].eq(
                targets[dataset.seven_rot]).sum().item() / \
                len(dataset.seven_rot)
    acc_1_rot = 100. * preds[dataset.one_rot].eq(
        targets[dataset.one_rot]).sum().item() / \
                len(dataset.one_rot)
    acc_7_rot_new = 100. * preds_new[dataset.seven_rot].eq(
        targets[dataset.seven_rot]).sum().item() / \
                len(dataset.seven_rot)
    acc_1_rot_new = 100. * preds_new[dataset.one_rot].eq(
        targets[dataset.one_rot]).sum().item() / \
                len(dataset.one_rot)
    print('Rotated Acc 7 {:.2f} new {:.2f}'.format(acc_7_rot, acc_7_rot_new))
    print('Rotated Acc 1 {:.2f} new {:.2f}'.format(acc_1_rot, acc_1_rot_new))
    acc_7_norot = 100. * preds[dataset.seven_norot].eq(
        targets[dataset.seven_norot]).sum().item() / \
                len(dataset.seven_norot)
    acc_1_norot = 100. * preds[dataset.one_norot].eq(
        targets[dataset.one_norot]).sum().item() / \
                len(dataset.one_norot)
    acc_7_norot_new = 100. * preds_new[dataset.seven_norot].eq(
        targets[dataset.seven_norot]).sum().item() / \
                    len(dataset.seven_norot)
    acc_1_norot_new = 100. * preds_new[dataset.one_norot].eq(
        targets[dataset.one_norot]).sum().item() / \
                    len(dataset.one_norot)
    print('No Rotated Acc 7 {:.2f} new {:.2f}'.format(acc_7_norot, acc_7_norot_new))
    print('No Rotated Acc 1 {:.2f} new {:.2f}'.format(acc_1_norot, acc_1_norot_new))
    plot_embedding(embed, targets, probs, acc, trans=False)
    plot_embedding(embed_new, targets, probs_new, acc_new, trans=True)



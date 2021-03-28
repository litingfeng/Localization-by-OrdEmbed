"""
Created on 2/23/2021 5:08 PM

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
def plot_embedding(embed, targets, probs, acc, dataset, trans=False):
    embed, probs = torch.cat(embed, dim=0), \
                            torch.cat(probs)
    embed, targets, probs = embed.cpu().numpy(), targets.cpu().numpy(), \
                            probs.cpu().numpy().flatten()
    less_confidents = np.argwhere(probs <= thresh).flatten()
    print('percent of less confidents {:.2f}%'.format(100. * len(less_confidents) / len(probs)))

    # embeddings
    four, one = np.argwhere(targets == 0).flatten(), np.argwhere(targets == 1).flatten()

    fig = plt.figure()
    plt.scatter(embed[four, 0], embed[four, 1], color='g', label='4', s=2)
    plt.scatter(embed[dataset.four_rot, 0], embed[dataset.four_rot, 1], color='b', label='4_rot', s=2, alpha=0.5)
    plt.scatter(embed[one, 0], embed[one, 1], color='r', label='1', s=2, alpha=0.5)
    plt.scatter(embed[dataset.one_rot, 0], embed[dataset.one_rot, 1], color='y', label='1_rot', s=2, alpha=0.5)
    #plt.scatter(embed[less_confidents, 0], embed[less_confidents, 1], color='c', label='less_conf', s=2, alpha=0.5)
    plt.legend()
    if not trans:
        plt.title('digit 4 1 Rotate 30% Epoch {} Acc {:.2f}'.format(epochs, acc))
        plt.savefig('./images/Rotate_embed.png')
    else:
        plt.title('Trans digit 4 1 Rotate 30% Epoch {} Acc {:.2f}'.format(epochs, acc))
        plt.savefig('./images/Rotate_trans_embed.png')
    plt.show()

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
    def __init__(self, root, subset, datapath=None, train=True, transform=None,
                 target_transform=None, download=True):
        super(myCIFAR, self).__init__(root, train=train, download=download,
                                      transform=transform, target_transform=target_transform)
        self.subset = subset
        phase = 'train' if train else 'test'
        if not datapath: #TODO sync with self-pcaed-classification-cifar.py
            # select 4, 1
            self.newdata, self.newtargets = [], []
            for idx, digit in enumerate(self.targets):
                if digit == 4 or digit == 1:
                    self.newdata.append(idx)
                    target = 0 if digit == 4 else 1
                    self.newtargets.append(target)
            # randomly sample ~100 for each class
            inds = np.random.permutation(len(self.newdata))
            self.newdata, self.newtargets = np.array(self.newdata)[inds[:100]], \
                                            np.array(self.newtargets)[inds[:100]]
            self.rot_inds = np.random.choice(100, size=30, replace=False) # select 100 to rotate
            print('rot number of 4 {}\nrot number of 1 {}'.format(
                len(np.where(self.newtargets[self.rot_inds] == 0)[0]),
                len(np.where(self.newtargets[self.rot_inds] == 1)[0])))
            self.norot_inds =np.array(list(set(range(100)) - set(self.rot_inds)))
            print('self.norot_inds ', self.norot_inds.shape)
            pickle.dump((self.newdata, self.newtargets, self.rot_inds, self.norot_inds),
                        open('{}_cifar_all_rot30_large.pkl'.format(phase), 'wb'))
        else:
            self.newdata, self.newtargets, self.rot_inds, self.norot_inds = \
                pickle.load(open('{}_cifar_all_rot30_large.pkl'.format(phase), 'rb'))
        print('number of 4 {}\nnumber of 1 {}'.format(len(np.where(self.newtargets == 0)[0]),
                                                      len(np.where(self.newtargets == 1)[0])))
        print(' rot: {} norot_inds: {} '.format(self.rot_inds.shape, self.norot_inds.shape))

        # select which subset to train
        if self.subset == 'original_all': # use all original(non-roated) 200 samples
            self.data = [self.data[i] for i in self.newdata]
            self.targets = self.newtargets
            self.data = np.stack(self.data)
        elif self.subset == 'original_half': # use 100 non-rotated samples
            print('self.norot_inds ', self.norot_inds.shape)
            self.data = [self.data[self.newdata[i]] for i in self.norot_inds]
            self.targets = self.newtargets[self.norot_inds]
            self.data = np.stack(self.data)
        elif self.subset == 'half_half': # use half rotated & half non-rotated
            self.orgdata = [self.data[i].transpose(2,0,1) for i in self.newdata] # HWC->CHW
            self.data = [ np.rot90(data.copy(), axes=(-2,-1))
                          if i in self.rot_inds else data
                          for i, data in enumerate(self.orgdata) ]
            self.data = np.stack(self.data).transpose(0, 2, 3, 1)
            #.transpose(0, 2, 3, 1)
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
        #index = self.rot_inds[0]
        data, target = self.data[index], self.targets[index]
        img = Image.fromarray(data)

        if DEBUG:
            print(index in self.rot_inds)
            f, ax = plt.subplots()
            ax.imshow(np.asarray(img), cmap='gray', interpolation='none')
            print('target ', target)
            plt.show()
            f, ax = plt.subplots()
            ax.imshow(np.asarray(
                Image.fromarray(self.orgdata[index].transpose(1,2,0))), cmap='gray', interpolation='none')
            print('target ', target)
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
    # # random.seed(seed)  # Python random module.
    # torch.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    #random_seed = 2
    thresh = 0.9
    # torch.backends.cudnn.enabled = False
    # torch.manual_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = myCIFAR(root='.', subset='half_half',
                      datapath='.',
                      train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]))

    test_set = myCIFAR(root='.', subset='original_all',
                       datapath='.',
                       train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]))
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=512, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=512, shuffle=False, num_workers=8)
    print('total images of 4 & 1: ', len(dataset), ' test: ', len(test_set))

    run_accs = {'train': [], 'test': []}
    for run in range(10):
        embedding_net = EmbeddingNet()
        net = Net(embedding_net, 2).to(device)
        agent = Agent(dim=5, num_action=2).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        #optimizer_agent = torch.optim.Adam(agent.parameters(), lr=0.001)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.1)
        #criterion2 = nn.CrossEntropyLoss(reduction='none')

        '''
        training
        '''
        net.train()
        total_loss, total_pol_losses, total_rewards = [], [], []
        total_r_back = []
        accs, accs_new = [], []
        less_confs, new_less_confs = [], []
        epochs = 10
        for epoch in range(epochs):
            train_losses, pol_losses, rewards = [], [], []
            r_back = []
            correct, correct_new = 0, 0
            for batch_idx, (data, target, index) in enumerate(train_loader):
                batch_size = data.shape[0]
                data, target = data.to(device), target.to(device)
                isrot = dataset.rot_dict[index].to(device)
                optimizer.zero_grad()

                # train classifier
                feature, output = net(data)
                loss = criterion(output, target)
                output = F.softmax(output, dim=1)
                prob, pred = output.max(1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tCls Loss: {:.4f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #            100. * batch_idx / len(train_loader),
            #     np.mean(train_losses)))

            total_loss.append(np.mean(train_losses))

            acc = 100. * correct / len(train_loader.dataset)

            accs.append(acc)
            # accs_new.append(acc_new)
            if epoch == 2:
                torch.save({
                    'epoch': epoch,
                    'state_dict': net.state_dict(),
                    'acc': acc
                }, 'half_half_large_2.pth.tar')
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'acc': acc
        }, 'half_half_large_9.pth.tar')

        # fig = plt.figure()
        # plt.plot(range(epochs), total_loss)
        # plt.legend(['Train Loss'], loc='upper right')
        # plt.xlabel('epoch')
        # plt.ylabel('loss')
        # plt.show()
        # fig = plt.figure()
        # plt.plot(range(epochs), accs, label='acc')
        # # plt.plot(range(epochs), accs_new, label='acc_new')
        # plt.legend()
        # plt.xlabel('epoch')
        # plt.ylabel('accuracy (%)')
        # plt.show()

        run_accs['train'].append(acc)

        '''
        testing
        '''
        ckpt = torch.load('half_half_large_9.pth.tar')
        net.load_state_dict(ckpt['state_dict'])
        print('loaded ckpt epoch {} acc {:.2f}%'.format(ckpt['epoch'], ckpt['acc']))
        acc = evaluate(net, test_loader)
        run_accs['test'].append(acc)
        exit()

    print('run accs\ntrain\n', run_accs['train'])
    print('\ntest\n', run_accs['test'])
    print('train acc: {:.2f}+-{:.2f}\n'
          'test acc: {:.2f}+-{:.2f}'.format(
        np.mean(run_accs['train']),
        np.std(run_accs['train']),
        np.mean(run_accs['test']),
        np.std(run_accs['test'])))

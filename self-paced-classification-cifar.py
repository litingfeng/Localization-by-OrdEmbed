"""
Created on 2/18/2021 7:18 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import os, wandb
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

DEBUG = False
# dataloader class
class myCIFAR(datasets.CIFAR10):
    def __init__(self, root, subset, datapath=None, train=True, transform=None,
                 target_transform=None, download=True):
        super(myCIFAR, self).__init__(root, train=train, download=download,
                                      transform=transform, target_transform=target_transform)
        self.subset = subset
        phase = 'train' if train else 'test'
        if not datapath:
            # select 4, 1
            self.newdata, self.newtargets = [], []
            for idx, digit in enumerate(self.targets):
                if digit == 4 or digit == 1:
                    self.newdata.append(idx)
                    target = 0 if digit == 4 else 1
                    self.newtargets.append(target)
            # randomly sample ~100 for each class
            #inds = np.random.permutation(len(self.newdata))
            # self.newdata, self.newtargets = np.array(self.newdata) #[inds[:100]], \
            #                                 np.array(self.newtargets)#[inds[:100]]
            self.newdata, self.newtargets = np.array(self.newdata), \
                                        np.array(self.newtargets)
            self.rot_inds = np.random.choice(len(self.newdata),
                                             size=int(0.3*len(self.newdata)), replace=False) # select 100 to rotate
            print('rot number of 4 {}\nrot number of 1 {}'.format(
                len(np.where(self.newtargets[self.rot_inds] == 0)[0]),
                len(np.where(self.newtargets[self.rot_inds] == 1)[0])))
            self.norot_inds =np.array(list(set(range(len(self.newdata))) - set(self.rot_inds)))
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
        elif self.subset == 'half_half': #TODO # use half rotated & half non-rotated
            self.data = [self.data[i] for i in self.newdata]
            self.data = [ np.rot90(data.copy(), axes=(-2,-1))
                          if i in self.rot_inds else data
                          for i, data in enumerate(self.data) ]
            print('data ', self.data[0].shape)
            self.data = np.stack(self.data)
            self.targets = self.newtargets
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
        #index = self.rot_inds[10]
        data, target = self.data[index], self.targets[index]
        img = Image.fromarray(data)

        if DEBUG:
            print(index in self.rot_inds)
            f, ax = plt.subplots()
            ax.imshow(np.asarray(img), cmap='gray', interpolation='none')
            print('target ', target)
            plt.show()
            exit()
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join('/research/cbim/vast/tl601/Dataset', 'MNIST', 'raw')

    @property
    def processed_folder(self):
        return os.path.join('/research/cbim/vast/tl601/Dataset', 'MNIST', 'processed')

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
        #print('output ', output.shape)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class EmbeddingNet_Simple(nn.Module):
    def __init__(self):
        super(EmbeddingNet_Simple, self).__init__()
        self.fc = nn.Sequential(nn.Linear(28*28, 64),
                                nn.PReLU(),
                                nn.Linear(64, 5)
                                )

    def forward(self, x):
        output = x.view(x.shape[0], -1)
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

if __name__ == "__main__":
    # seed = 2
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # np.random.seed(seed)  # Numpy module.
    # #random.seed(seed)  # Python random module.
    # torch.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # random_seed = 2
    # torch.backends.cudnn.enabled = False
    # torch.manual_seed(random_seed)
    mode = 'original_half'

    #os.environ['WANDB_NAME'] = "binary_classification"
    #wandb.init(project="selfpaced")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = myCIFAR(root = '.', subset = mode,
                      #datapath='.',
                      train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]))
    test_set = myCIFAR(root='.', subset='original_all',
                      #datapath='.',
                      train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]))

    train_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=512, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_set,
                                               batch_size=512, shuffle=False, num_workers=8)
    print('total images of 4 & 1: ', len(dataset))

    run_accs = {'train': [], 'test': []}
    for run in range(10):
        embedding_net = EmbeddingNet()
        net = Net(embedding_net, 2).to(device)
        criterion = nn.CrossEntropyLoss()

        #wandb.watch(net)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)#, momentum=0.5)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[13], gamma=0.1)
        criterion = nn.CrossEntropyLoss()

        '''
        training
        '''
        net.train()
        total_loss = []
        accs = []
        epochs = 15
        for epoch in range(epochs):
            train_losses = []
            correct = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                feature, output = net(data)
                loss = criterion(output, target)
                output = F.softmax(output, dim=1)
                _, pred = output.max(1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

                loss.backward()
                optimizer.step()
                #scheduler.step()
                train_losses.append(loss.item())

            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #            100. * batch_idx / len(train_loader), np.mean(train_losses)))

            total_loss.append(np.mean(train_losses))
            acc = 100. * correct / len(train_loader.dataset)
            accs.append(acc)
            torch.save({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'acc': acc
            }, '{}_rot30_large.pth.tar'.format(mode))
            #break
        #plot loss and acc
        # fig = plt.figure()
        # plt.plot(range(epochs), total_loss)
        # plt.legend(['Train Loss'], loc='upper right')
        # plt.xlabel('epoch')
        # plt.ylabel('loss')
        # plt.show()
        # fig = plt.figure()
        # plt.plot(range(epochs), accs)
        # plt.legend(['Train Acc'], loc='upper right')
        # plt.xlabel('epoch')
        # plt.ylabel('accuracy (%)')
        # plt.show()

        run_accs['train'].append(acc)

        '''
        testing
        '''
        ckpt = torch.load('{}_rot30_large.pth.tar'.format(mode))
        net.load_state_dict(ckpt['state_dict'])
        print('loaded ckpt epoch {} acc {:.2f}%'.format(ckpt['epoch'], ckpt['acc']))
        # plot embedding
        net.eval()
        correct = 0
        embed, targets, probs = [], [], []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                feature, output = net(data)
                output = F.softmax(output, dim=1)
                prob, pred = output.max(1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                embed.append(feature)
                targets.append(target)
                probs.append(prob)
        acc = 100. * correct / len(test_loader.dataset)
        print('acc in eval mode: {:.2f}%'.format(acc))
        run_accs['test'].append(acc)
        embed, targets, probs = torch.cat(embed, dim=0), torch.cat(targets), \
                            torch.cat(probs)
        embed, targets, probs = embed.cpu().numpy(), targets.cpu().numpy(), \
                                probs.cpu().numpy().flatten()
        #less_confidents = np.argwhere(probs <= 0.9).flatten()
        #print('percent of less confidents {:.2f}%'.format(100. * len(less_confidents) / len(probs)))


    print('run accs\ntrain\n', run_accs['train'])
    print('\ntest\n', run_accs['test'])
    print('train acc: {:.2f}+-{:.2f}\n'
          'test acc: {:.2f}+-{:.2f}'.format(
        np.mean(run_accs['train']),
        np.std(run_accs['train']),
        np.mean(run_accs['test']),
        np.std(run_accs['test'])))
    # # histogram of probs
    # fig = plt.figure()
    # plt.hist(probs, density=True)
    # plt.title('histogram of probs')
    # plt.show()

    # # embeddings
    # four, one = np.argwhere(targets==0).flatten(), np.argwhere(targets==1).flatten()
    # print('four ', four.shape, ' one ', one.shape)
    # print('four_rot ', len(dataset.four_rot), ' one_rot ', len(dataset.one_rot))
    #
    # fig = plt.figure()
    # plt.scatter(embed[four, 0], embed[four, 1], color='g', label='7', s=2)
    # plt.scatter(embed[dataset.four_rot, 0], embed[dataset.four_rot, 1], color='b', label='7_rot', s=2, alpha=0.5)
    # plt.scatter(embed[one, 0], embed[one, 1], color='r', label='1', s=2, alpha=0.5)
    # plt.scatter(embed[dataset.one_rot, 0], embed[dataset.one_rot, 1], color='y', label='1_rot', s=2, alpha=0.5)
    # #plt.scatter(embed[less_confidents, 0], embed[less_confidents, 1], color='c', label='less_conf', s=2, alpha=0.5)
    # plt.legend()
    # plt.title('digit 7 1 Rotate 30% Epoch {} Acc {:.2f}'.format(epochs, acc))
    # plt.show()
    #
    # fig = plt.figure()
    # plt.scatter(embed[four, 0], embed[four, 1], color='g', label='7', s=2)
    # plt.scatter(embed[dataset.four_rot, 0], embed[dataset.four_rot, 1], color='b', label='7_rot', s=2, alpha=0.5)
    # plt.scatter(embed[one, 0], embed[one, 1], color='r', label='1', s=2, alpha=0.5)
    # plt.scatter(embed[dataset.one_rot, 0], embed[dataset.one_rot, 1], color='y', label='1_rot', s=2, alpha=0.5)
    # plt.scatter(embed[less_confidents, 0], embed[less_confidents, 1], color='c', label='less_conf', s=2, alpha=0.5)
    # plt.legend()
    # plt.title('digit 7 1 Rotate 30% Epoch {} Acc {:.2f}'.format(epochs, acc))
    # plt.show()




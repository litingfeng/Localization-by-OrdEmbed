"""
Created on 3/2/2021 12:10 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import imageio, wandb, os, argparse
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform
        print('data ', self.data.shape, ' target ', self.target.shape)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)

class Trans_Net(nn.Module):
    def __init__(self):
        super(Trans_Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 13),
            #nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--step', default=8, type=int,
                        metavar='N', help='scheduler step')
    parser.add_argument('--optimizer', type=str, default="SGD",
                        choices=['Adam', 'SGD'])
    parser.add_argument('-b', '--batch_size', default=512, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ['WANDB_NAME'] = 'all_epoch_data'
    wandb.init(project="selfpaced")
    # each is a list of 15 (epochs)
    # losses_ij[0]: (10000,3), loss_0, loss_i, loss_j
    # feat_ijs[0]: (10000, 15), feature_0, feature_i, feature_j
    # save_ts[0]: (10000, 2), a1, a2
    losses_ij, save_ts, feat_ijs = pickle.load(open('losses_ij.pkl', 'rb'))

    trans_net = Trans_Net().to(device)
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(trans_net.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(trans_net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.1)
    #criterion = nn.MSELoss()
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    wandb.watch(trans_net)
    wandb.config.update(args)


    # view last epoch
    x = [torch.cat(losses_ij[i]) for i in range(len(losses_ij))]
    f = [torch.cat(feat_ijs[i]) for i in range(len(losses_ij))]
    y = [torch.cat(save_ts[i]).cpu() for i in range(len(losses_ij))]
    x = torch.cat(x)
    f = torch.cat(f)
    y = torch.cat(y)

    round_y = torch.round(y[:, 0]*10**3)/ 10**3
    classes = {v: k for k,v in
               enumerate(set(list(round_y.numpy())))}
    print('classes ', len(classes), classes)

    # plt.figure()
    x_sub = x[:, 2] - x[:, 1]
    y_sub = y[:, 1] - y[:, 0]
    # print('x_sub ', x_sub.shape)
    # plt.plot(x_sub.data.numpy(), label='x_sub')
    # plt.plot(y_sub.data.numpy(), label = 'y_sub')
    # plt.legend()
    # plt.title('loss_j - loss_i vs. a2 - a1')
    # plt.show()
    #
    # plt.figure()
    with torch.no_grad():
        feat_0, feat_i, feat_j = f[:, :5], f[:, 5:10], f[:, 10:]
        d0i, d0j = F.pairwise_distance(feat_0, feat_i), F.pairwise_distance(feat_0, feat_j)
        f_sub = d0j - d0i
    # y_sub = y[:, 1] - y[:, 0]
    # #plt.plot((d0j-d0i).data.numpy(), label='d_sub')
    # plt.plot(y_sub.data.numpy(), label='y_sub')
    # plt.legend()
    # plt.title('d0j - d0i vs. a2 - a1')
    # plt.show()
    #exit()

    niter = 0
    epochs = 100
    tri = torch.tril(torch.ones(len(classes), len(classes)))
    # normalize input
    with torch.no_grad():
        # use loss_i, a1
        data = x[:, 1].float() - x[:, 0].float()
        target = torch.round(y[:, 0]*10**3)/ 10**3
        target = [classes[target[i].item()] for i in range(target.shape[0])]
        # convert to multi-hot
        target = [tri[t] for t in target]
        target = torch.stack(target)

        # use loss_j, a2
        data2 = x[:, 2].float() - x[:, 0].float()
        target2 = torch.round(y[:, 1] * 10 ** 3) / 10 ** 3
        target2 = [classes[target2[i].item()] for i in range(target2.shape[0])]
        # convert to multi-hot
        target2 = [tri[t] for t in target2]
        target2 = torch.stack(target2)
        print('data ', data.shape)
        data, target = torch.cat((data, data2)), torch.cat((target, target2))
        print('all data ', data.shape, target.shape)
        mean, std = torch.mean(data).item(), torch.mean(data).item()
        dataset = MyDataset(data.view(300000,1, 1, 1), target,
                            transform=transforms.Normalize(mean=(mean,),
                                         std=(std, )))
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8
        )

    #for epoch, (loss_ijs, ts) in enumerate(zip(losses_ij, save_ts)):
    total_loss, accs = [], []
    for epoch in range(epochs):
        train_loss = []
        correct = 0
        scheduler.step()
        for batch_idx, (loss_ij, ang) in enumerate(loader):

            optimizer.zero_grad()
            loss_ij, ang = loss_ij.view(-1, 1).to(device), ang.to(device)
            output = trans_net(loss_ij.view(-1, 1))

            loss = criterion(output, ang)
            # output = F.softmax(output, dim=1)
            # _, pred = output.max(1, keepdim=True)
            output = F.sigmoid(output)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            correct += (output == ang).all(dim=1).sum().item()
            # print('output ', output[:2], '\nang ', ang[:2])
            # exit()
            #correct += pred.eq(ang.view_as(pred)).sum().item()

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
        total_loss.append(np.mean(train_loss))

        acc = 100. * correct / target.shape[0]
        accs.append(acc)
        current_lr = scheduler.get_last_lr()[0]
        log_message = {"Loss": total_loss[-1],
                       "Acc": acc,
                       'learning rate': current_lr}

        wandb.log(log_message, step=epoch)

    # fig = plt.figure()
    # plt.plot(range(epochs), total_loss)
    # plt.legend(['Train Loss'], loc='upper right')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.show()
    # fig = plt.figure()
    #
    # print('accs ', accs)
    # plt.plot(range(epochs), accs)
    # plt.legend(['Train Acc'], loc='upper right')
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy (%)')
    # plt.show()
    # """
    # test
    # """
    # with torch.no_grad():
    #     test_loss = []
    #     for batch_idx in range(20):
    #         if batch_idx != 19:
    #             loss_ij, t = data[batch_idx*512: (batch_idx+1)*512], \
    #                          target[batch_idx*512: (batch_idx+1)*512]
    #         else:
    #             loss_ij, t = data[batch_idx*512:], y_sub[batch_idx*512:]
    #
    #         loss_ij, t = loss_ij.view(-1, 1).to(device), t.to(device)
    #         output = trans_net(loss_ij.view(-1, 1))
    #         loss = criterion(output, t)
    #         # print('loss_ij ', loss_ij.view(-1)[:10])
    #         # print('output ', output.view(-1)[:10])
    #         # print('exp_t ', torch.exp(0.1*t)[:10].view(-1))
    #
    #         test_loss.append(loss.item())
    #
    #     print('test_loss: {:.2f}+-{:.2f}'.format(np.mean(test_loss), np.std(test_loss)) )



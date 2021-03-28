"""
Created on 2/18/2021 7:18 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import os, wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms
from opts import parser
from datasets.cifar import myCIFAR
from models.regression_model import Net, EmbeddingNet
from transform import UnNormalize

DEBUG = False
unorm = UnNormalize(mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010))

def init_dataloader():

    train_set = myCIFAR(root='.', subset='half_half', ratio=0.5,
                        angles=np.array(range(0, 181, args.angle_step)),
                        datapath='.', train=True,
                      transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                     (0.2023, 0.1994, 0.2010)),
                            ]))

    test_set = myCIFAR(root='.', subset='original_all', ratio=0.5,
                       angles=np.array(range(0, 181, args.angle_step)),
                       datapath='.', train=False,
                       transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                     (0.2023, 0.1994, 0.2010)),
                            ]))
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size,
                                               shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=args.batch_size,
                                              shuffle=False, num_workers=8)
    print('total images of 4 & 1: ', len(train_set), ' test: ', len(test_set))

    return train_loader, test_loader

def init_model(n_a_classes):
    embedding_net = EmbeddingNet()
    net = Net(embedding_net, 2, n_a_classes, mode='classify').to(device)
    # if load pretrained classifier
    if args.pretrained != '':
        ckpt = torch.load(os.path.join('/research/cbim/vast/tl601/results/'
                          'selfpaced/ordinal/',
                          args.pretrained))
        net.load_state_dict(ckpt['state_dict'])
        print('loaded from {}\nckpt epoch {} acc {:.2f}%'.format(args.pretrained,
                                                    ckpt['epoch'], ckpt['acc']))
    if args.freeze == 1: # freeze angle net, only train classifier
        # for name, value in net.named_parameters():
        #     print('net name ', name)
        for name, value in net.angle_net.named_parameters():
            value.requires_grad = False

    return net

def feed_data(model, data_loader, is_train):
    if is_train:
        phase = 'Train'
        model.train()
    else:
        phase = 'Test'
        model.eval()

    thresh = args.thresh
    total_loss, losses_i, losses_order = [], [], []
    correct_cls, correct_ang = 0, 0
    len_dataloader = len(data_loader)
    for batch_idx, (xis, target, ai, angle, index) in enumerate(data_loader):
        batch_size = xis.shape[0]
        ai, xis, target = ai.to(device),xis.to(device), \
                          target.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            feature, output, output_o = model(xis)
            _, pred_order = output_o.max(1)
            loss_order = criterion_order(output_o, ai)

        # rotate and forward
        # print('a ', ai, ' xi ', xis.shape)
        # print('angle ', angle)
        # print('pred_order ', pred_order)
        imgs = []
        for img, a in zip(xis, pred_order):
            #print('angle ', a, -idx2cls[a.item()])
            if DEBUG:
                img_o = unorm(img.clone())
                img_o = transforms.ToPILImage()(img_o)

            img = TF.rotate(img, -float(idx2cls[a.item()]))
            imgs.append(img)
            if DEBUG:
                img_t = unorm(img)
                img_t = transforms.ToPILImage()(img_t)
                f, ax = plt.subplots()
                ax.imshow(np.asarray(img_o), interpolation='none')
                plt.title('img_o')
                plt.show()
                f, ax = plt.subplots()
                ax.imshow(np.asarray(img_t), interpolation='none')
                plt.title('img_t')
                plt.show()
                exit()
        imgs = torch.stack(imgs)
        feature, output, output_o = model(imgs)

        # compute loss
        loss_i = criterion(output, target) # (bs,)

        loss = loss_i
        if phase == 'Train':
            loss.backward()
            optimizer.step()

        # compute accuracy
        output = F.softmax(output, dim=1)
        _, pred = output.max(1, keepdim=True)
        correct_cls += pred.eq(target.view_as(pred)).sum().item()

        correct_ang += (pred_order == ai).sum().item()

        total_loss.append(loss.item())
        losses_i.append(loss_i.mean().item())
        losses_order.append(loss_order.item())
        if batch_idx % args.log_interval == 0:
            if batch_idx == 0:
                print('\n')
            print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                phase, epoch, batch_idx * batch_size,
                len(data_loader.dataset),
                              100. * batch_idx / len_dataloader, np.mean(total_loss)))

    acc = 100. * correct_cls / len(data_loader.dataset)
    acc_o = 100. * correct_ang / len(data_loader.dataset)

    log_message = {"{} Total Loss".format(phase): np.mean(total_loss),
                   "{} Cls Loss_i".format(phase): np.mean(losses_i),
                   "{} Order Loss".format(phase): np.mean(losses_order),
                   '{} Cls Acc'.format(phase): acc,
                   '{} Angle Acc'.format(phase): acc_o,
                   }
    if phase == 'Train':
        current_lr = scheduler.get_last_lr()[0]
        log_message.update({'learning rate': current_lr})
    wandb.log(log_message, step=epoch)

    return total_loss, acc_o

if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ['WANDB_NAME'] = args.savename
    wandb.init(project="selfpaced")
    saveroot = os.path.join('/research/cbim/vast/tl601/results/selfpaced/ordinal',
                            args.savename)
    os.makedirs(saveroot, exist_ok=True)

    train_loader, test_loader = init_dataloader()
    idx2cls = train_loader.dataset.idx2cls

    net = init_model(n_a_classes=len(train_loader.dataset.angles))
    wandb.watch(net, log_freq=10)
    wandb.config.update(args)

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.1)

    criterion = nn.CrossEntropyLoss()
    criterion_order = nn.CrossEntropyLoss()

    if args.evaluate  == 1:
        epoch = 0
        test_loss, test_acc = feed_data(net,  data_loader=train_loader, is_train=False)
        print('test order acc ', test_acc)
        exit()

    best_acc = 0.
    save_model = False
    no_improve_epoch = 0
    for epoch in range(args.epochs):
        train_log = feed_data(net, data_loader=train_loader, is_train=True)
        scheduler.step()

        with torch.no_grad():
            test_loss, test_acc = feed_data(net,  data_loader=test_loader, is_train=False)

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

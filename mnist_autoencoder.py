"""
Train autoencoder on digit 4 of clutter mnist
https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
Created on 1/7/2021 10:49 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import os
from opts import parser
import wandb
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.optim as optim
from torchvision import transforms
from util.augmentations import Compose
from util.data_aug import Resize
from datasets.clutter_mnist_scale_ae import MNIST_CoLoc
from models.mnist_ae import Autoencoder
import matplotlib.pyplot as plt
from util.utils import generate_boxes
from util.utils import convert_image_np

DEBUG = False

args = parser.parse_args()
n_epochs = args.epochs
batch_size_train = args.batch_size
batch_size_test = 16
learning_rate = args.lr
log_interval = 10

os.environ['WANDB_NAME'] = args.savename
wandb.init(project="selfpaced")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kwargs = {'num_workers': 8, 'pin_memory': True}

train_transform = transforms.Compose([
    #transforms.RandomCrop(84, padding=4),
    # transforms.RandomResizedCrop(84),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(degrees=(-90,90)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
# train_transform = Compose([Resize(84)])
# test_transform = Compose([Resize(84)])
trainset = MNIST_CoLoc(root='.', train=True, digit=args.digit,
                       sample_size=args.sample_size, bg_name=args.bg_name,
                       datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                       clutter=1, transform=train_transform
                       )
testset = MNIST_CoLoc(root='.', train=False, digit=args.digit,
                      bg_name=args.bg_name,
                      datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                      clutter=1, transform=test_transform
                      )

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size_train, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size_test, shuffle=True, **kwargs)
print('total train image: ', len(train_loader.dataset), ' test image: ',
      len(test_loader.dataset))

def train(model, num_epochs=5):
    criterion = nn.MSELoss() # mean square error loss
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                 milestones=args.steps, gamma=0.1)

    #outputs = []

    loss_all = []
    for epoch in range(num_epochs):
        total_loss = []
        scheduler.step()
        for img in train_loader:
            img = img.to(device)
            optimizer.zero_grad()
            recon = model(img)
            loss = criterion(recon, img)
            #loss = F.smooth_l1_loss(recon, img)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())

            if DEBUG:
                print('output ', recon[:3])

        total_loss = np.mean(total_loss)
        loss_all.append(total_loss)
        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, total_loss))

        input = convert_image_np(
            torchvision.utils.make_grid(img[:16].cpu(), nrow=4,
                                        padding=5, pad_value=123), norm=False)
        output = convert_image_np(
            torchvision.utils.make_grid(recon[:16].cpu().detach(), nrow=4,
                                        padding=5, pad_value=123), norm=False)
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(input)
        axarr[0].set_title('Dataset Images')
        axarr[0].set_axis_off()

        axarr[1].imshow(output)
        axarr[1].set_title('Reconstructed Images')
        axarr[1].set_axis_off()
        #plt.show()
        current_lr = scheduler.get_last_lr()[0]

        log_message = {"Loss": total_loss,
                       'learning rate': current_lr,
                       "Images": wandb.Image(plt)}
        wandb.log(log_message, step=epoch)
        plt.close()
        #outputs.append((epoch, img, recon),)
    torch.save(model.state_dict(),
               os.path.join(saveroot, 'last.pth.tar'))

    return loss_all

def test(model, data_loader, is_train, upload=True):
    model.eval()

    count = 0
    correct_inst, correct_proto, correct_inst_other, correct_proto_other = 0, 0, 0, 0
    len_dataloader = len(data_loader)
    for batch_idx, (data, target, ti, tj, ioui, iouj) in enumerate(data_loader):
        batch_size = data.shape[0]
        # if batch_size < args.samples_per_class:
        #     continue
        data, target, ti, tj, ioui, iouj = data.to(device), target.to(device).float(), \
                                           ti.to(device), tj.to(device), \
                                           ioui.to(device), iouj.to(device)

        '''split to support set and target set '''
        num = batch_size % args.samples_per_class
        if num != 0:
            # print('batch_size ', batch_size)
            # print('num ', num)
            data, target, ti, tj, ioui, iouj = data[:-num], target[:-num], \
                                               ti[:-num], tj[:-num], \
                                               ioui[:-num], iouj[:-num]
            batch_size = data.shape[0]
        count += batch_size
        num_group = batch_size // args.samples_per_class
        rnd_inds = np.random.permutation(num_group)

        # rois
        rois_0 = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                            target.to(device)), dim=1)
        rois_i = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                            ti.to(device)), dim=1)
        rois_j = torch.cat((torch.arange(0, batch_size).float().view(-1, 1).to(device),
                            tj.to(device)), dim=1)

        recon_img0, pooled_0 = model(data, rois_0) # (bs, 1, 84, 84), (bs, 1600)
        recon_imgi, pooled_i = model(data, rois_i)
        recon_imgj, pooled_j = model(data, rois_j)

        pooled_0 = pooled_0.view(batch_size, -1)
        pooled_i = pooled_i.view(batch_size, -1)
        pooled_j = pooled_j.view(batch_size, -1)

        # compute loss order
        iou = torch.stack((ioui, iouj), 1)  # (bs, 2)
        f = torch.stack((pooled_i, pooled_j)).permute(1, 0, 2)  # (bs, 2, dim)

        inds = torch.max(iou, 1).indices
        p = f[torch.arange(0, batch_size), inds, :].view(-1,
                                        args.samples_per_class, 3136)[rnd_inds].view(-1, 3136)
        n = f[torch.arange(0, batch_size), 1 - inds, :].view(-1,
                                        args.samples_per_class, 3136)[rnd_inds].view(-1, 3136)
        #proto = torch.mean(pooled_0, dim=0).view(1, -1).repeat(batch_size, 1)
        # proto should be the mean of each group
        proto = pooled_0.view(-1, args.samples_per_class, 3136).mean(1) # (num_group, 512)
        proto = proto.unsqueeze(1).repeat(1, args.samples_per_class, 1).view(-1, 3136)

        # compute accuracy self
        d_a0a1 = F.pairwise_distance(pooled_0, pooled_i)
        d_a0a2 = F.pairwise_distance(pooled_0, pooled_j)
        sub_pix = iouj - ioui
        sub_d = d_a0a1 - d_a0a2
        mask = (torch.sign(sub_d) == torch.sign(sub_pix))
        correct_inst += mask.sum().item()

        d_a0a1 = F.pairwise_distance(proto, pooled_i)
        d_a0a2 = F.pairwise_distance(proto, pooled_j)
        sub_d = d_a0a1 - d_a0a2
        mask = (torch.sign(sub_d) == torch.sign(sub_pix))
        correct_proto += mask.sum().item()
         # acc other
        pooled_i = pooled_i.view(-1, args.samples_per_class, 3136)[rnd_inds].view(-1, 3136)
        pooled_j = pooled_j.view(-1, args.samples_per_class, 3136)[rnd_inds].view(-1, 3136)
        ioui = ioui.view(-1, args.samples_per_class)[rnd_inds].view(-1)
        iouj = iouj.view(-1, args.samples_per_class)[rnd_inds].view(-1)

        d_a0a1 = F.pairwise_distance(pooled_0, pooled_i)
        d_a0a2 = F.pairwise_distance(pooled_0, pooled_j)
        sub_pix = iouj - ioui
        sub_d = d_a0a1 - d_a0a2
        mask = (torch.sign(sub_d) == torch.sign(sub_pix))
        correct_inst_other += mask.sum().item()

        d_a0a1 = F.pairwise_distance(proto, pooled_i)
        d_a0a2 = F.pairwise_distance(proto, pooled_j)
        sub_d = d_a0a1 - d_a0a2
        mask = (torch.sign(sub_d) == torch.sign(sub_pix))
        correct_proto_other += mask.sum().item()

        if batch_idx % args.log_interval == 0:
            if batch_idx == 0:
                print('\n')
            print('Test: [{}/{} ({:.0f}%)]'.format(
                batch_idx * batch_size,
                len(data_loader.dataset),
                100. * batch_idx / len_dataloader))

        #break
    print('count ', count, ' len(data) ', len(data_loader.dataset))
    acc_inst = 100. * correct_inst / count #len(data_loader.dataset)
    acc_proto = 100. * correct_proto / count #len(data_loader.dataset)
    acc_inst_other = 100. * correct_inst_other / count #len(data_loader.dataset)
    acc_proto_other = 100. * correct_proto_other / count #len(data_loader.dataset)

    # plot
    input = convert_image_np(
        torchvision.utils.make_grid(data[:16].cpu(), nrow=4,
                                    padding=5, pad_value=123), norm=False)
    recon_img0 = TF.normalize(recon_img0, (0.0334, 0.0301, 0.0279), (0.1653, 0.1532, 0.1490))
    #recon_img0 = TF.normalize(recon_img0, (0.1307,), (0.3081,))
    output = convert_image_np(
        torchvision.utils.make_grid(recon_img0[:16].cpu().detach(), nrow=4,
                                    padding=5, pad_value=123),
            norm=False)
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(input)
    axarr[0].set_title('Dataset Images')
    axarr[0].set_axis_off()

    axarr[1].imshow(output)
    axarr[1].set_title('Reconstructed Images')
    axarr[1].set_axis_off()

    phase = 'Test'
    log_message = {
                   '{} Ord Acc Inst'.format(phase): acc_inst,
                   '{} Ord Acc Proto'.format(phase): acc_proto,
                   '{} Ord Acc Inst Other'.format(phase): acc_inst_other,
                   '{} Ord Acc Proto Other'.format(phase): acc_proto_other,
                   "{} Images".format(phase): wandb.Image(plt)
                   }

    plt.close()

    return acc_inst, acc_proto, acc_inst_other, acc_proto_other

def evaluate(upload=True):
    from datasets.clutter_mnist_scale_anchor import MNIST_CoLoc
    anchors = generate_boxes(base_size=4,
                             feat_height=21, feat_width=21, img_size=84,
                             feat_stride=4,
                             ratios=[1.0],
                             min_box_side=28,
                             scales=np.array(range(7, 20)))
    kwargs = {'num_workers': 8, 'pin_memory': True}
    test_transform = Compose([Resize(84)])
    for name in ['last']:
        model_path = os.path.join(saveroot, '{}.pth.tar'.format(name))
        # load model
        print('loading net ckpt from ', model_path)
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt)
        # acc_proto = ckpt['acc']
        # print(("=> loaded model checkpoint epoch {} {}\n\tacc {:.2f}".format(epoch, name, acc_proto)))
        test_accs_inst, test_accs_proto, test_accs_inst_other, \
        test_accs_proto_other = [], [], [], []
        with torch.no_grad():
            for d in range(0, 10):
                if d != 4:
                    continue
                testset = MNIST_CoLoc(root='.', train=False, digit=d, anchors=anchors,
                                      datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                                      clutter=1, transform=test_transform
                                      )
                test_loader = torch.utils.data.DataLoader(
                    testset, batch_size=args.batch_size, shuffle=False, **kwargs)
                test_acc_inst, test_acc_proto, test_acc_inst_other, \
                test_acc_proto_other = test(model,
                                            data_loader=test_loader, is_train=False, upload=False)
                test_accs_inst.append(test_acc_inst)
                test_accs_proto.append(test_acc_proto)
                test_accs_inst_other.append(test_acc_inst_other)
                test_accs_proto_other.append(test_acc_proto_other)
                print(d, ' test_acc_inst {:.2f}\ttest_acc_proto {:.2f}'
                         '\ttest_acc_inst_other {:.2f}\ttest_acc_proto_other {:.2f}'.format(
                      test_acc_inst, test_acc_proto, test_acc_inst_other,
                      test_acc_proto_other))
            print('mean test_acc_inst {:.2f}\ttest_acc_proto {:.2f}'
                     '\ttest_acc_inst_other {:.2f}\ttest_acc_proto_other {:.2f}'.format(
                np.mean(test_accs_inst), np.mean(test_accs_proto), np.mean(test_accs_inst_other),
                np.mean(test_accs_proto_other)))
        if upload:
            wandb.run.summary.update({
            'Test {} Ord Acc Inst'.format(name.capitalize()): acc_inst,
            'Test {} Ord Acc Proto'.format(name.capitalize()): acc_proto,
            'Test {} Ord Acc Inst Other'.format(name.capitalize()): acc_inst_other,
            'Test {} Ord Acc Proto Other'.format(name.capitalize()): acc_proto_other,
            'Test {} Gen Ord Acc Inst'.format(name.capitalize()): np.mean(test_accs_inst),
            'Test {} Gen Ord Acc Proto'.format(name.capitalize()): np.mean(test_accs_proto),
            'Test {} Gen Ord Acc Inst Other'.format(name.capitalize()): np.mean(test_accs_inst_other),
            'Test {} Gen Ord Acc Proto Other'.format(name.capitalize()): np.mean(test_accs_proto_other),
            })

model = Autoencoder(channel=1).to(device)
wandb.watch(model, log_freq=10)
saveroot = os.path.join('/research/cbim/vast/tl601/results/mnist/',
                       args.savename)
os.makedirs(saveroot, exist_ok=True)
loss_all = train(model, num_epochs=n_epochs)

# args.sample_size = '50'
# network_state_dict = torch.load(os.path.join('/research/cbim/vast/tl601/results/mnist/',
#                                 'ae_50_lr1e03_step50_2/last.pth.tar'))
# from models.mnist_ae_ord import Autoencoder
# model = Autoencoder(pooling_mode=args.pooling_mode,
#                     pooling_size=args.pooling_size).to(device)
# model.load_state_dict(network_state_dict)
# evaluate()

# data = next(iter(test_loader))
# output = model(data.to(device))
# #data = TF.normalize(data, (0.1307,), (0.3081,))
# output = TF.normalize(output, (0.1307,), (0.3081,))
# input = convert_image_np(
#             torchvision.utils.make_grid(data, nrow=4, padding=5, pad_value=123), norm=False)
# output = convert_image_np(
#     torchvision.utils.make_grid(output.cpu().detach(),nrow=4, padding=5, pad_value=123), norm=False)
# f, axarr = plt.subplots(1, 2)
# axarr[0].imshow(input)
# axarr[0].set_title('Dataset Images')
# axarr[0].set_axis_off()
# axarr[1].imshow(output)
# axarr[1].set_title('Reconstructed Images')
# axarr[1].set_axis_off()
# plt.show()


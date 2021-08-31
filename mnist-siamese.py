"""
Train a siamese/triplet network for mnist
https://github.com/adambielski/siamese-triplet/blob/master/Experiments_MNIST.ipynb
Created on 5/23/2021 6:47 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import os, torch, wandb
from opts import parser
import numpy as np
from MulticoreTSNE import MulticoreTSNE as TSNE
import torch.nn.functional as F
from util.augmentations import Compose
from util.data_aug import Resize
from torchvision import transforms
from datasets.batch_sampler import BalancedBatchSampler
import matplotlib.pyplot as plt
from util.utils import generate_boxes
from datasets.mnist_siamese import MNIST_CoLoc
from datasets.clutter_mnist_scale_anchor_2digit_2p import MNIST_CoLoc
from models.mnist_embed import EmbeddingNet
from model.roi_layers import ROIPool
from losses.contrastive import OnlineContrastiveLoss
from losses.selectors import AllPositivePairSelector, HardNegativePairSelector

DEBUG = False

# def init_dataloader():
#     # 28 28 images
#     kwargs = {'num_workers': 8, 'pin_memory': True}
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])
#     trainset = MNIST_CoLoc(root='.', train=True, digit1=args.digit,
#                            sample_size=args.sample_size,
#                            digit2=args.digit2, transform=transform)
#     testset = MNIST_CoLoc(root='.', train=False, digit1=args.digit,
#                           sample_size=args.sample_size,
#                           digit2=args.digit2, transform=transform)
#
#     train_batch_sampler = BalancedBatchSampler(trainset.targets, n_classes=2, n_samples=25)
#     test_batch_sampler = BalancedBatchSampler(testset.targets, n_classes=2, n_samples=25)
#
#     train_loader = torch.utils.data.DataLoader(
#         trainset, batch_sampler=train_batch_sampler, **kwargs)
#     test_loader = torch.utils.data.DataLoader(
#         testset, batch_sampler=test_batch_sampler, **kwargs)
#     train_loader_org = torch.utils.data.DataLoader(
#         trainset, batch_size=args.batch_size, shuffle=False, **kwargs)
#     test_loader_org = torch.utils.data.DataLoader(
#         testset, batch_size=args.batch_size, shuffle=False, **kwargs)
#     print('total train image: ', len(train_loader.dataset), ' test image: ',
#           len(test_loader.dataset))
#     return train_loader, test_loader, train_loader_org, test_loader_org
def init_dataloader():
    kwargs = {'num_workers': 8, 'pin_memory': True}

    train_transform = Compose([Resize(84)])
    test_transform = Compose([Resize(84)])
    # for 28*28
    anchors = generate_boxes(base_size=4,
                             feat_height=21, feat_width=21, img_size=84,
                             feat_stride=4,
                             ratios=[1.0],
                             min_box_side=28,
                             scales=np.array(range(7, 20)))

    print('number of anchors for 84*84 image ', anchors.shape[0])
    trainset = MNIST_CoLoc(root='.', train=True, digit=args.digit,
                           anchors=anchors, bg_name=args.bg_name,
                           sample_size=args.sample_size,
                           datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                           clutter=1, transform=train_transform,
                           )
    testset = MNIST_CoLoc(root='.', train=False, digit=args.digit,
                          anchors=anchors, bg_name=args.bg_name,
                          datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                          clutter=1, transform=test_transform
                          )

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    print('total train image: ', len(train_loader.dataset), ' test image: ',
          len(test_loader.dataset))
    return train_loader, test_loader

def init_model():
    embedding_net = EmbeddingNet(dim=args.dim).to(device)
    return embedding_net

def feed_data(model, data_loader, is_train):
    if is_train:
        phase = 'Train'
        model.train()
    else:
        phase = 'Test'
        model.eval()

    losses = []
    len_dataloader = len(data_loader)
    for batch_idx, (data, target, _, _) in enumerate(data_loader):
        batch_size = data.shape[0]
        data, target = data.to(device),  target.to(device).float()
        if phase == 'Train':
            optimizer.zero_grad()

        rois_targets = torch.cat((torch.arange(0, batch_size).unsqueeze(1).repeat(1, 2).view(-1, 1).to(device),
                                  target.view(-1, 5)[:, :4]), dim=1)  # (2bs, 5)
        roi_region = roi_pool(data, rois_targets) # (2bs, 1, 28, 28)

        outputs = model(roi_region)
        loss = loss_fn(outputs, target.view(-1, 5)[:, -1])

        losses.append(loss.item())
        if phase == 'Train':
            loss.backward()
            optimizer.step()

        if batch_idx % args.log_interval == 0:
            if batch_idx == 0:
                print('\n')
            print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                phase, epoch, batch_idx * batch_size,
                len(data_loader.dataset),
                100. * batch_idx / len_dataloader, np.mean(losses)))
        #break

    losses = np.mean(losses)
    log_message = {"{} Loss".format(phase): losses}

    if phase == 'Train':
        current_lr = scheduler.get_last_lr()[0]
        log_message.update({'learning rate': current_lr})

    wandb.log(log_message, step=epoch)

    return losses

def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset) * 2, 2))
        embeddings_org = np.zeros((len(dataloader.dataset) * 2, args.dim))
        labels = np.zeros(len(dataloader.dataset)* 2)
        k = 0
        for images, target, _, _ in dataloader:
            batch_size = images.shape[0]
            images = images.to(device)
            rois_targets = torch.cat((torch.arange(0, batch_size).unsqueeze(1).repeat(1, 2).view(-1, 1),
                                      target.view(-1, 5)[:, :4].float()), dim=1).to(device)  # (2bs, 5)
            roi_region = roi_pool(images, rois_targets)  # (2bs, 1, 28, 28)
            embed_org = model.get_embedding(roi_region).data.cpu().numpy()
            embed = TSNE(n_jobs=4).fit_transform(embed_org)
            embeddings[k:(k+len(images)*2)] = embed
            embeddings_org[k:(k+len(images)*2)] = embed_org
            labels[k:(k+len(images)*2)] = target.view(-1,5)[:,-1].numpy()
            k += len(images)*2

    return embeddings, embeddings_org, labels

def plot_embeddings(embeddings, targets, xlim=None, ylim=None, set='train'):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
    mnist_classes = [str(args.digit), str(args.digit2)]
    plt.figure(figsize=(10,10))
    for i in range(10):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes)
    log_message= {"{} Images".format(set): wandb.Image(plt)}
    wandb.log(log_message)
    plt.close()


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['WANDB_NAME'] = args.savename
    wandb.init(project="selfpaced")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saveroot = os.path.join('/research/cbim/vast/tl601/results/mnist/',
                            args.savename)
    os.makedirs(saveroot, exist_ok=True)

    train_loader, test_loader = init_dataloader()

    roi_pool = ROIPool((28, 28), 1.0)
    model = init_model()
    wandb.watch(model, log_freq=10)
    wandb.config.update(args)

    loss_fn = OnlineContrastiveLoss(args.margin_cls, HardNegativePairSelector())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=args.steps, gamma=0.1)

    if args.evaluate == 1:
        # loaded siamese pretrained net
        model_path = os.path.join(saveroot, 'last.pth.tar')
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['state_dict'])

    for epoch in range(args.epochs):
        train_log = feed_data(model, data_loader=train_loader, is_train=True)
        scheduler.step()

        with torch.no_grad():
            test_loss = feed_data(model, data_loader=test_loader, is_train=False)

        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
        }, os.path.join(saveroot, 'last.pth.tar'))

    train_embeddings_ocl, train_embeds, train_labels_ocl = extract_embeddings(train_loader, model)
    plot_embeddings(train_embeddings_ocl, train_labels_ocl, set='train')
    val_embeddings_ocl, val_embeds, val_labels_ocl = extract_embeddings(test_loader, model)
    plot_embeddings(val_embeddings_ocl, val_labels_ocl, set='test')

    # distances
    pos_embed, neg_embed = train_embeds[np.where(train_labels_ocl == args.digit)], \
                           train_embeds[np.where(train_labels_ocl == args.digit2)]
    print('train proto dists ', F.pairwise_distance(torch.from_numpy(pos_embed).mean(0).unsqueeze(0),
                                                    torch.from_numpy(neg_embed).mean(0).unsqueeze(0)).item())
    pos_embed, neg_embed = val_embeds[np.where(val_labels_ocl == args.digit)], \
                           val_embeds[np.where(val_labels_ocl == args.digit2)]
    print('test proto dists ', F.pairwise_distance(torch.from_numpy(pos_embed).mean(0).unsqueeze(0),
                                                   torch.from_numpy(neg_embed).mean(0).unsqueeze(0)).item())












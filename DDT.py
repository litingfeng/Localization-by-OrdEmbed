"""
Created on 5/27/2021 8:19 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import os, wandb, cv2
import torch, copy
import numpy as np
import torchvision
import scipy.linalg as la
import torch.nn as nn
from util.utils import box_iou
from matplotlib.patches import Rectangle
import torch.nn.functional as F
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from transform import MyBoxScaleTransform
from util.data_aug import *
from util.bbox_util import *
from util import joint_transforms as t
from util.utils import box_iou
from util.augmentations import Compose
from opts import parser
from datasets.clutter_mnist_scale_rl import MNIST_CoLoc
from datasets.cub_fewshot_ddt import CUB_fewshot
from models.mnist_scale_model import Net_DDT
from util.utils import convert_image_np
from constants import IMAGENET_MEAN, IMAGENET_STD
import torchvision.models as models
import matplotlib.cm as mpl_color_map
from PIL import Image, ImageFilter

DEBUG = True

def init_dataloader():
    kwargs = {'num_workers': 8, 'pin_memory': True}

    if args.dataset == 'mnist':
        train_transform = Compose([Resize(84)])
        test_transform = Compose([Resize(84)])
        trainset = MNIST_CoLoc(root='.', train=True, digit=args.digit,
                               sample_size=args.sample_size, bg_name=args.bg_name,
                               datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                               clutter=1, transform=train_transform)
        testset = MNIST_CoLoc(root='.', train=False, digit=args.digit,
                              bg_name=args.bg_name,
                              datapath='/research/cbim/vast/tl601/Dataset/Synthesis_mnist',
                              clutter=1, transform=test_transform)
    elif args.dataset == 'cub':
        transform = t.Compose([
            t.ConvertFromPIL(),
            t.ToPercentCoords(),
            t.Resize(args.img_size),
            t.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            t.ToTensor()  # no change to (0, 1)
        ])
        trainset = CUB_fewshot('/research/cbim/vast/tl601/Dataset/CUB_200_2011',
                               mode='base', img_size=args.img_size, loose=args.loose,
                               transform=transform)
        testset = CUB_fewshot('/research/cbim/vast/tl601/Dataset/CUB_200_2011',
                              mode='val', img_size=args.img_size, loose=args.loose,
                              transform=transform)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    print('total train image: ', len(train_loader.dataset), ' test image: ',
          len(test_loader.dataset))
    return train_loader, test_loader

def init_model():
    if args.dataset == 'mnist':
        net = Net_DDT()
        # if load pretrained encoder
        if args.pretrained != '':
            ckpt = torch.load(os.path.join('/research/cbim/vast/tl601/results/'
                                           'mnist/', args.pretrained))
            if 'state_dict' in ckpt.keys():
                mnist_pretrained_dict = ckpt['state_dict']
            else:
                mnist_pretrained_dict = ckpt
            embeddingnet_dict = net.state_dict()

            filtered_dict = {}
            for k, v in mnist_pretrained_dict.items():
                if k in embeddingnet_dict:
                    print('load ', k)
                    filtered_dict[k] = v
            embeddingnet_dict.update(filtered_dict)
            net.load_state_dict(embeddingnet_dict)
            if 'state_dict' in ckpt.keys():
                if 'acc_inst' in ckpt.keys():
                    print('loaded from {}\nckpt epoch {} acc {:.2f}%'.format(args.pretrained,
                                                                             ckpt['epoch'], ckpt['acc_inst']))
                else:
                    print('loaded from {}\nckpt epoch {} acc {:.2f}%'.format(args.pretrained,
                                                                             ckpt['epoch'], ckpt['acc']))
            else:
                print('loaded pretained MNIST encoder ', args.pretrained)
    elif args.dataset == 'cub':
        model_path = '/research/cbim/vast/tl601/projects/selfpaced/vgg16_caffe.pth'
        vgg = models.vgg16()
        # use pretrained model
        print("Loading pretrained weights from %s" % (model_path))
        state_dict = torch.load(model_path)
        vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})

        net = nn.Sequential(*list(vgg.features._modules.values())[:-1])

    if args.freeze == 1:
        for name, value in net.named_parameters():
            value.requires_grad = False
        net.eval()

    return net

def cov(m, rowvar=True, inplace=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    if inplace:
        m -= torch.mean(m, dim=1, keepdim=True)
    else:
        m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()

def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    print('no_trans_heatmap ', no_trans_heatmap.shape)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image

def feed_data(model, data_loader):
    correct = 0
    len_dataloader = len(data_loader)

    count = 0
    dim = 64 if args.dataset == 'mnist' else 512
    wh = 42 if args.dataset == 'mnist' else 14
    scale = 2. if args.dataset == 'mnist' else 16
    embeds = torch.zeros((len(data_loader.dataset), dim, wh*wh)).to(device)
    targets = []
    for batch_idx, (data, target) in enumerate(data_loader):
        batch_size = data.shape[0]
        data, target = data.to(device), target.float()
        targets.append(target)

        with torch.no_grad():
            embed = model(data) # (bs, 64, 42, 42)

        embeds[count: (count+batch_size)] = embed.view(batch_size, dim, -1)
        count += batch_size

    targets = torch.cat(targets)
    embeds = embeds.permute(1, 0, 2).reshape(dim, -1).permute(1,0).to(device) # (42*42*len(dataset), 64)

    covmat = cov(embeds, rowvar=False) # (64, 64)
    eigvals, eigvecs = la.eig(covmat.cpu().numpy())
    sorted_indexes = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indexes]   # (64,)
    eigvecs = eigvecs[:, sorted_indexes] # (64, 64)
    eig_top = torch.from_numpy(eigvecs[:, 0]).to(device) # (64, )

    '''predict'''
    mean =  torch.mean(embeds, dim=0).squeeze() #(64,)
    p_ij = eig_top.unsqueeze(0).matmul((embeds -  mean).t()) # [1, 42*42*len(dataset)]
    p_ij = p_ij.reshape(len(data_loader.dataset), 1, wh, wh) # [len(dataset), 1, 42, 42]
    P_ij = F.interpolate(p_ij, scale_factor=scale, mode='nearest').cpu().numpy() # (len(dataset), 1, 84, 84)

    '''plot'''
    data, target = next(iter(data_loader))
    input = convert_image_np(
        torchvision.utils.make_grid(data[:4].cpu(), nrow=2,
                                    padding=5, pad_value=123), norm=False)
    f, axarr = plt.subplots()
    axarr.imshow(input)
    axarr.set_title('Dataset Images')
    axarr.set_axis_off()

    correct = 0
    for k, img in enumerate(P_ij):
        img = img.transpose(1,2,0)
        thresh = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY)[1]

        #print(k, ' img ', img.shape, np.min(img), np.max(img), '\n', np.min(thresh), ' ', np.max(thresh))
        # apply connected component analysis to the thresholded image
        output = cv2.connectedComponentsWithStats(
            np.uint8(thresh*255), 4, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output

        max_x, max_y, max_w, max_h = 0, 0, 0, 0
        max_area = 0
        # loop over the number of unique connected component labels
        for i in range(0, numLabels):
            # if this is the first component then we examine the
            # *background* (typically we would just ignore this
            # component in our loop)
            if i == 0:
                text = "examining component {}/{} (background)".format(
                    i + 1, numLabels)
            # otherwise, we are examining an actual connected component
            else:
                text = "examining component {}/{}".format(i + 1, numLabels)
            # print a status message update for the current connected
            # component

            # extract the connected component statistics and centroid for
            # the current label
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]
            #print("[INFO] {} area: {}".format(text, area))
            if area > max_area:
                max_x, max_y, max_w, max_h = x, y, w, h
                max_area = area
                break

        if args.dataset == 'mnist':
            input = convert_image_np(data[0].repeat(3,1,1), norm=False).astype(np.uint8)
        else:
            input = convert_image_np(data[0], norm=False).astype(np.uint8)
        print('input ', input.shape)
        activation_map = np.clip(img, 0, 1).reshape(args.img_size, args.img_size)*255.

        # # Get Heatmap
        # heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)
        # # Zero out
        # heatmap[np.where(activation_map >= 100)] = 0
        #
        # out = cv2.addWeighted(src1=input, alpha=0.8, src2=heatmap, beta=0.4, gamma=0)
        # out = cv2.resize(out, dsize=(500, 500))
        #
        # plt.imshow(out[:, :, ::-1])
        # plt.show()

        # print('activation_map ', activation_map.shape)
        # heatmap, heatmap_on_image = apply_colormap_on_image(Image.fromarray(input),
        #                                                     activation_map, 'hsv')
        # print('heatmap ', type(heatmap), ' heatonimg ', type(heatmap_on_image))

        f, axarr = plt.subplots()
        axarr.imshow(np.array(heatmap))
        axarr.set_title('heatmap')
        plt.show()

        f, axarr = plt.subplots()
        axarr.imshow(np.array(heatmap_on_image))
        axarr.set_title('heatmap_on_image')
        plt.show()
        #axarr.set_axis_off()

        # patch = Rectangle((max_x, max_y), max_w, max_h, linewidth=1,
        #                   edgecolor='r', facecolor='none', fill=False)
        # axarr.add_patch(patch)
        # plt.title('{}'.format(k))
        # plt.show()

        if k == 0:
            exit()
        # calculate accuracy

        pred_boxes = torch.tensor([[max_x, max_y, max_x+max_w-1, max_y+max_h-1]], dtype=torch.float32)
        iou = torch.diagonal(box_iou(pred_boxes,
                                     targets[k].unsqueeze(0)), 0)
        #print('iou ', iou)
        correct += (iou >= 0.5).sum()

    acc = 100. * correct / len(data_loader.dataset)
    print('acc ', acc)

if __name__ == '__main__':
    args = parser.parse_args()
    args.dataset = 'cub'

    #os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.sample_size = 50
    train_loader, test_loader = init_dataloader()

    embed_net = init_model()
    embed_net = nn.DataParallel(embed_net).to(device)
    feed_data(embed_net, train_loader)
# -*- coding: utf-8 -*-
# @Time : 9/7/21 2:39 PM
# @Author : Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--lr_ag', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--evaluate', default=False, action='store_true',
                    help='evaluate model')
parser.add_argument('--train_mode', default='shuffle_proto', type=str,
                    choices=['self', 'proto', 'shuffle_self',
                             'shuffle_proto'],
                    help='training mode of embedding net or agent')
parser.add_argument('--bg_name', type=str, default="impulse_noise",
                    choices=['clean','clutter', 'patch', 'gaussian_noise', 'impulse_noise',
                             'warbler', 'gull', 'wren', 'sparrow', 'oriole', 'kingfisher', 'vireo'],
                    help='type of background of dataset')
parser.add_argument('--dataset', default='mnist', type=str,
                    choices=['mnist', 'cub', 'coco'])
parser.add_argument('--savename', default='', type=str,
                    help='saving name')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--optimizer', type=str, default="Adam",
                    choices=['Adam', 'SGD'])
parser.add_argument('--optimizer_ag', type=str, default="Adam",
                    choices=['Adam', 'SGD', 'RMS'])
parser.add_argument('--digit', default=4, type=int,
                    help='digit to train on')
parser.add_argument('--patience', default=50, type=int,
                    help='number of epochs to tolerate the no improvement of val acc')
parser.add_argument('--lamb', default=0.1, type=float,
                    help='weight of loss order')
parser.add_argument('--lamb_base', default=1., type=float,
                    help='weight of base loss')
parser.add_argument('--lamb_ent', default=0.5, type=float,
                    help='weight of loss entropy')
parser.add_argument('--margin', default=0., type=float,
                    help='margin of loss')
parser.add_argument('--sample_size', default='whole', type=str,
                    #choices=['50', '100', '300', '600', '1k', '2k', '3k', '4k','whole'],
                    help='Sample size of train set, choice from')
parser.add_argument('-b', '--batch_size', default=512, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('--steps', type=int, nargs='+',
                    help='steps of multistep scheduler')
parser.add_argument('--steps_ag', default=50, type=int,
                    help='scheduler step')
parser.add_argument('--pooling_size', default=7, type=int,
                    help='size after roi pooling or align')
parser.add_argument('--pooling_mode', default='align',
                    help='align | pool')
parser.add_argument('--log_interval', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--samples_per_class', default=20, type=int,
                    help='samples per class in support and target set')
parser.add_argument('--support_size', default=5, type=int,
                    help='number of images in support set ')
parser.add_argument("--rnn", default=True, action='store_false',
                    help="if true, use rnn")
parser.add_argument("--hidden_size", type=int, default=48,
                    help="hidden size of rnn or hidden layer")
parser.add_argument('--pretrained', default='', type=str,
                    help='pretrained model name')
parser.add_argument('--pretrained_agent', default='', type=str,
                    help='pretrained agent model name')
parser.add_argument('--backbone', default='', type=str,
                    help='backbone for cub dataset')
parser.add_argument("--freeze", default=True, action='store_false',
                    help="freeze encoder")
parser.add_argument("--img_size", type=int, default=84,
                    help="image size")
parser.add_argument("--min_box_side", type=int, default=28,
                    help="mininum box size")
parser.add_argument("--sign", default=False, action='store_true',
                    help="if use sign of reward")
parser.add_argument('--seq_len', default=5, type=int,
                    help='number of steps that agent could take')
parser.add_argument('--num_act', default=5, type=int,
                    help='number of actions')
parser.add_argument('--gamma', default=0.99, type=float,
                    help='discount factor')
parser.add_argument('--dim', default=512, type=int,
                    help='dimension of projection head')
parser.add_argument('--dim_ag', default=64, type=int,
                    help='dimension of projection head')

# for coco
parser.add_argument('--sel_cls', nargs='+', type=str,
                    help='list of selected classes ')

"""
Created on 3/7/2021 3:00 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_ag', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', default=0, type=int,
                    help='evaluate model')
parser.add_argument('--savename', default='', type=str,
                    help='saving name')
parser.add_argument('--digit', default=4, type=int,
                    help='digit to train on')
parser.add_argument('--pretrained', default='', type=str,
                    help='pretrained model name')
parser.add_argument('--seq_len', default=5, type=int,
                    help='number of steps that agent could take')
parser.add_argument('--freeze', default=1, type=int,
                    help='freeze classifier')
parser.add_argument('--sparse', default=0, type=int,
                    help='if use sparse reward')
parser.add_argument('--sign', default=0, type=int,
                    help='if use sign of reward')
parser.add_argument("--hidden_size", type=int, default=None,
                    help="hidden size of rnn, if None, no rnn")
parser.add_argument('--angle_step', default=90, type=int,
                    help='angle of each rotation for both dataset and actor')
parser.add_argument('--gamma', default=0.99, type=float,
                        help='discount factor')
parser.add_argument('--step', default=50, type=int,
                    metavar='N', help='scheduler step')
parser.add_argument('--steps', type=int, nargs='+',
                    help='steps of multistep scheduler')
parser.add_argument('--step_ag', default=50, type=int,
                    metavar='N', help='scheduler step')
parser.add_argument('--norm', default=0, type=int,
                    help='if 1 then norm feature')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--optimizer', type=str, default="Adam",
                    choices=['Adam', 'SGD'])
parser.add_argument('--optimizer_ag', type=str, default="Adam",
                    choices=['Adam', 'SGD'])
parser.add_argument('--margin', default=0., type=float,
                    help='margin of loss')
parser.add_argument('-b', '--batch_size', default=512, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lamb', default=1.0, type=float,
                    help='weight of loss order')
parser.add_argument('--lamb_ent', default=1.0, type=float,
                    help='weight of entropy loss')
parser.add_argument('--lamb_int', default=0.0, type=float,
                    help='weight of intrinsic loss')
parser.add_argument('--thresh', default=0.001, type=float,
                    help='threshold of loss for self-paced learning')
parser.add_argument('--log_interval', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--patience', default=50, type=int,
                    help='number of epochs to tolerate the no improvement of val acc')

#============== order regression =======================
parser.add_argument('--mode', default='cum_feat', type=str,
                    help='use feature or loss, '
                         'cum_feat | cum_loss | classify')
parser.add_argument('--pooling_size', default=28, type=int,
                    help='size after roi pooling or align')
parser.add_argument('--pooling_mode', default='align',
                    help='align | pool')

# =================== agent ===========================
parser.add_argument('--num_act', default=5, type=int,
                    help='number of actions')
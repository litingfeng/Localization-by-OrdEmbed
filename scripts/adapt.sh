#!/bin/bash
# Adapt localization agent on corrupted MNIST, from random patch to clutter, digit 3 to 2.
CUDA_VISIBLE_DEVICES=0 python adapt_agent.py \
                       --savename adapt_mnist_randpatch2clutter \
                       --pretrained_agent agent_mnist_ae_randpatch/best.pth.tar \
                       --pretrained pretrain_mnist_ae_randpatch/last.pth.tar \
                       --bg_name clutter --digit 2 --num_act 10 --batch_size 512

# Adapt localization agent on CUB dataset, from warbler to wren.
CUDA_VISIBLE_DEVICES=0 python adapt_agent.py \
                       --savename adapt_cub_warbler2wren \
                       --pretrained_agent agent_vgg_cub_warb15/best.pth.tar \
                       --pretrained pretrain_vgg_cub_warb15/last.pth.tar \
                       --bg_name wren --dataset cub --backbone vgg16 \
                       --dim 1024 --dim_ag 512 --img_size 224 \
                       --min_box_side 40 --batch_size 64

# Adapt localization agent on COCO dataset, from dog to cat.
CUDA_VISIBLE_DEVICES=0 python adapt_agent.py \
                       --savename adapt_coco_dog2cat \
                       --pretrained_agent agent_vgg_coco_dog/best.pth.tar \
                       --pretrained pretrain_vgg_coco_dog/last.pth.tar \
                       --dataset coco --backbone vgg16 --sel_cls cat \
                       --dim 1024 --dim_ag 512 --img_size 224 \
                       --min_box_side 40 --batch_size 64
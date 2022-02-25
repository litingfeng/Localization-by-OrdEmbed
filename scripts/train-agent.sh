#!/bin/bash
# Train localization agent on corrupted MNIST
CUDA_VISIBLE_DEVICES=0 python train_agent.py \
                       --savename agent_mnist_ae_randpatch \
                       --pretrained pretrain_mnist_ae_randpatch/best.pth.tar \
                       --img_size 84 --bg_name patch --sample_size 50 \
                       --digit 3 --num_act 10 --batch_size 50 --steps_ag 200

# Train localization agent on cub dataset
CUDA_VISIBLE_DEVICES=0 python train_agent.py \
                       --savename agent_vgg_cub_warb15 \
                       --pretrained pretrain_vgg_cub_warb15/last.pth.tar \
                       --dataset cub --dim 1024 --dim_ag 512 \
                       --bg_name warbler --backbone vgg16 --img_size 224 \
                       --min_box_side 40 --batch_size 50

# Train localization agent on coco dataset
CUDA_VISIBLE_DEVICES=0 python train_agent.py \
                       --savename agent_vgg_coco_dog \
                       --pretrained pretrain_vgg_coco_dog/last.pth.tar \
                       --dataset coco --backbone vgg16 \
                       --dim 1024 --dim_ag 512 --sel_cls dog \
                       --img_size 224 --min_box_side 40 --batch_size 50

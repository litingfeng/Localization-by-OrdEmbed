#!/bin/bash
# pretrain RoI encoder on corrupted MNIST
CUDA_VISIBLE_DEVICES=0 python pretrain_encoder_ordinal.py \
                         --savename pretrain_mnist_ae_randpatch \
                         --digit 3 --bg_name patch --samples_per_class 10 \
                         --sample_size 50 --batch_size 50

# pretrain RoI encoder on cub dataset
CUDA_VISIBLE_DEVICES=0 python pretrain_encoder_ordinal.py \
                        --savename pretrain_vgg_cub_warb15 \
                        --dataset cub --backbone vgg16 --bg_name warbler \
                        --dim 1024 --batch_size 50 --img_size 224 --lamb 1.0

# pretrain RoI encoder on coco dataset
CUDA_VISIBLE_DEVICES=0 python pretrain_encoder_ordinal.py \
                        --savename pretrain_vgg_coco_dog \
                        --dataset coco --sel_cls dog --backbone vgg16 \
                        --dim 1024 --batch_size 75 --img_size 224 --lamb 1.0
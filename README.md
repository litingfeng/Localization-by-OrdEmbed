# Learning Transferable Reward for Query Object Localization with Policy Adaptation
This repository is the official implementation of our ICLR 2022 paper "Learning Transferable Reward for Query Object Localization with Policy Adaptation".

![framework](https://github.com/litingfeng/Localization-by-Ordinal-Embedding/blob/main/images/fig1.png)

## Requirements
- Python3
- Pytorch >= 1.5
- [Weights & Biases](https://www.wandb.com/)
- sklearn
- [Add RoI pool/align layer](https://medium.com/@andrewjong/how-to-use-roi-pool-and-roi-align-in-your-neural-networks-pytorch-1-0-b43e3d22d073)
- [ImageNet pretrained vgg16 model](https://drive.google.com/file/d/1gVx7Ye8HsTJZHyEChfFQBtc36PHS0xeO/view?usp=sharing)

## Corrupted MNIST
### Data Preparation
To generate corrupted MNIST data with variant background (4 background supported: *clean, clutter, patch, gaussian_noise, impulse_noise*). E.g., to generate patched MNIST, run the following code:
```
python datasets/generate_data.py --bg_name patch
```
### Training
1. Pretrain RoI encoder and projection head. Below is an example of training on digit 3.
```shell
CUDA_VISIBLE_DEVICES=0 python pretrain_encoder_ordinal.py \
                         --savename pretrain_mnist_ae_randpatch \
                         --digit 3 --bg_name patch --samples_per_class 10 \
                         --sample_size 50 --batch_size 50
```
2. Train localization agent.
```shell
CUDA_VISIBLE_DEVICES=0 python train_agent.py \
                       --savename agent_mnist_ae_randpatch \
                       --pretrained pretrain_mnist_ae_randpatch/best.pth.tar \
                       --img_size 84 --bg_name patch --sample_size 50 \
                       --digit 3 --num_act 10 --batch_size 50 --steps_ag 200
```
3. Test-time adaptation.
```shell
CUDA_VISIBLE_DEVICES=0 python adapt_agent.py \
                       --savename adapt_mnist_randpatch2clutter \
                       --pretrained_agent agent_mnist_ae_randpatch/best.pth.tar \
                       --pretrained pretrain_mnist_ae_randpatch/last.pth.tar \
                       --bg_name clutter --digit 2 --num_act 10 --batch_size 512
```
## CUB Dataset
### Data Preparation
Below is an example of generate `gull_59_64.json`. Other files are located at `datasets/cub_files`.
```
python datasets/generate_cub_filelist.py
```
### Training
- pretrain RoI encoder and projection head on warbler
```shell
CUDA_VISIBLE_DEVICES=0 python pretrain_encoder_ordinal.py \
                        --savename pretrain_vgg_cub_warb15 \
                        --dataset cub --backbone vgg16 --bg_name warbler \
                        --dim 1024 --batch_size 50 --img_size 224 --lamb 1.0
```
- train agent
```shell
CUDA_VISIBLE_DEVICES=0 python train_agent.py \
                       --savename agent_vgg_cub_warb15 \
                       --pretrained pretrain_vgg_cub_warb15/last.pth.tar \
                       --dataset cub --dim 1024 --dim_ag 512 \
                       --bg_name warbler --backbone vgg16 --img_size 224 \
                       --min_box_side 40 --batch_size 50
```
- adapt agent from warbler to wren
```shell
CUDA_VISIBLE_DEVICES=0 python adapt_agent.py \
                       --savename adapt_cub_warbler2wren \
                       --pretrained_agent agent_vgg_cub_warb15/best.pth.tar \
                       --pretrained pretrain_vgg_cub_warb15/last.pth.tar \
                       --bg_name wren --dataset cub --backbone vgg16 \
                       --dim 1024 --dim_ag 512 --img_size 224 \
                       --min_box_side 40 --batch_size 64
```

## COCO Dataset
### Training 
- pretrain RoI encoder and projection head on dog
```shell
CUDA_VISIBLE_DEVICES=0 python pretrain_encoder_ordinal.py \
                        --savename pretrain_vgg_coco_dog \
                        --dataset coco --sel_cls dog --backbone vgg16 \
                        --dim 1024 --batch_size 75 --img_size 224 --lamb 1.0
```
- train agent
```shell
CUDA_VISIBLE_DEVICES=0 python train_agent.py \
                       --savename agent_vgg_coco_dog \
                       --pretrained pretrain_vgg_coco_dog/last.pth.tar \
                       --dataset coco --backbone vgg16 \
                       --dim 1024 --dim_ag 512 --sel_cls dog \
                       --img_size 224 --min_box_side 40 --batch_size 50
```
- adapt agent from dog to cat
```shell
CUDA_VISIBLE_DEVICES=0 python adapt_agent.py \
                       --savename adapt_coco_dog2cat \
                       --pretrained_agent agent_vgg_coco_dog/best.pth.tar \
                       --pretrained pretrain_vgg_coco_dog/last.pth.tar \
                       --dataset coco --backbone vgg16 --sel_cls cat \
                       --dim 1024 --dim_ag 512 --img_size 224 \
                       --min_box_side 40 --batch_size 64
```
##Bibtex
```
@inproceedings{
li2022learning,
title={Learning Transferable Reward for Query Object Localization with Policy Adaptation},
author={Tingfeng Li and Shaobo Han and Martin Renqiang Min and Dimitris N. Metaxas},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=92tYQiil17}
}
```

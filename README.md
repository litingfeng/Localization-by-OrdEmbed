# Learning Transferable Reward for Query Object Localization with Policy Adaptation
This repository is the official implementation of our paper "Learning Transferable Reward for Query Object Localization with Policy Adaptation".

![framework](https://github.com/litingfeng/Localization-by-Ordinal-Embedding/blob/main/images/Fig1_v3.png)
**TODO**: 
1. include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials.
2. Add roi layer 

## Requirements
- Python3
- Pytorch >= 1.5
- [Weights & Biases](https://www.wandb.com/)
- sklearn

## Corrupted MNIST
### Data Preparation
To generate corrupted MNIST data with variant background (4 background supported: *clean, clutter, patch, gaussian_noise, impulse_noise*). E.g., to generate patched MNIST, run the following code:
```
python datasets/generate_data.py --bg_name patch
```
### Training
1. Pretrain RoI encoder and projection head. Below is an example of training on digit 3.
```shell
CUDA_VISIBLE_DEVICES=7 python pretrain_encoder_ordinal.py \
                         --savename pretrain_mnist_encoder \
                         --digit 3 \
                         --bg_name patch \
                         --lamb 0.1 \
                         --margin 60 \
                         --samples_per_class 10 \
                         --sample_size 50 \
                         --batch_size 50 \
                         --patience 150 \
                         --epochs 150 \
                         --steps 80
```
2. Train localization agent.
```shell
CUDA_VISIBLE_DEVICES=6 python train_agent.py \
                       --savename pretrain_mnist_agent \
                       --pretrained pretrain_mnist_encoder/best.pth.tar \
                       --img_size 84 \
                       --bg_name patch \
                       --samples_per_class 5 \
                       --sample_size 50 \
                       --digit 3 \
                       --hidden_size 48 \
                       --seq_len 10 \
                       --num_act 10 \
                       --patience 100 \
                       --epochs 250 \
                       --batch_size 50 \
                       --steps_ag 200
```
3. Test-time adaptation.
```shell
CUDA_VISIBLE_DEVICES=6 python adapt_agent.py \
                       --savename adapt_mnist_agent_3to2 \
                       --pretrained_agent pretrain_mnist_agent/best.pth.tar \
                       --pretrained pretrain_mnist_encoder/best.pth.tar \
                       --support_size 5 \
                       --bg_name clutter \
                       --sample_size whole \
                       --digit 2 \
                       --seq_len 10 \
                       --margin 60 \
                       --hidden_size 48 \
                       --num_act 10 \
                       --patience 50 \
                       --epochs 100 \
                       --batch_size 512 \
                       --steps_ag 80
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
                        --savename pretrain_cub_encoder \
                        --dataset cub \
                        --backbone vgg16 \
                        --bg_name warbler \
                        --dim 1024 \
                        --batch_size 50 \
                        --img_size 224 \
                        --log_interval 25 \
                        --samples_per_class 25 \
                        --lamb 1.0 \
                        --epochs 150 \
                        --margin 60 \
                        --patience 120 \
                        --steps 80
```
- train agent
```shell
CUDA_VISIBLE_DEVICES=7 python train_agent.py \
                       --savename pretrain_cub_agent \
                       --pretrained pretrain_cub_encoder/last.pth.tar \
                       --dataset cub \
                       --dim 1024 \
                       --dim_ag 512 \
                       --bg_name warbler \
                       --backbone vgg16 \
                       --img_size 224 \
                       --min_box_side 40 \
                       --samples_per_class 5 \
                       --seq_len 10 \
                       --hidden_size 48 \
                       --num_act 14 \
                       --patience 100 \
                       --log_interval 100 \
                       --epochs 150 \
                       --batch_size 50 \
                       --steps_ag 80
```
- adapt agent from warbler to gull
```shell
CUDA_VISIBLE_DEVICES=7 python adapt_agent.py \
                       --savename adapt_cub_agent_warbler2gull \
                       --pretrained_agent pretrain_cub_agent/best.pth.tar \
                       --pretrained pretrain_cub_encoder/last.pth.tar \
                       --evaluate \
                       --bg_name gull \
                       --dataset cub \
                       --backbone vgg16 \
                       --dim 1024 \
                       --dim_ag 512 \
                       --img_size 224 \
                       --support_size 5 \
                       --min_box_side 40 \
                       --seq_len 10 \
                       --hidden_size 48 \
                       --num_act 14 \
                       --patience 100 \
                       --log_interval 10 \
                       --epochs 150 \
                       --batch_size 64 \
                       --steps_ag 80

```
### Evaluation
### Pre-trained Models
### Results

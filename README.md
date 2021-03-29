# Localization-by-Ordinal-Embedding
This repo documents experiments steps and scripts used to localize objects with
ordinal embedding. 

## Updates
- __3/28/2021__: Add experiments for fixed size cluttered mnsit

## Requirements
- Python3
- Pytorch 1.5
- [Weights & Biases](https://www.wandb.com/)
- sklearn

## Fixed Size Cluttered MNIST
### Description
The problem for this experiment is to localize 28\*28 digit in 84\*84 cluttered 
background. Like the image shown below. 

<img src="https://github.com/litingfeng/Localization-by-Ordinal-Embedding/blob/main/images/example1.png" width="800" height="400">

There are five different images, where the groundtruth bounding box is annotated with green
box in the last column. We use reinforcement learning method to tackle this problem. The agent
takes up to ten steps to localize the target, starting from the entire image. It takes two steps
to achieve this.
- __Learn ordinal embedding__: train an ordinal embedding network which has the property that the closer the box to 
groundtruth box, the closer the distance in embedding space. The closeness in pixel space is measured by IoU, and in 
embedding space is by L2 distance. Note that different from regular object detection method, this 
method doesn't require classification during training, thus is able to do localization when there's only 
one class/digit in train set. And it can generalize to other digits without finetuning. To learn this, we use embedding net shown below,  
<img src="https://github.com/litingfeng/Localization-by-Ordinal-Embedding/blob/main/images/embedingnet.png" width="800" height="160">
where
<img src="https://github.com/litingfeng/Localization-by-Ordinal-Embedding/blob/main/images/note.png" width="800" height="120">

- __Joint train with agent__: After training embedding network, we jointly train it with the agent. 
 The reward is defined as 
 <p align="center">
   <img src="https://github.com/litingfeng/Localization-by-Ordinal-Embedding/blob/main/images/reward.png" width="300" height="30">
</p>
And there are 10 actions (the last red dot means stay):  
<img src="https://github.com/litingfeng/Localization-by-Ordinal-Embedding/blob/main/images/action.png" width="800" height="100">

### Train scripts
- __Pretrain embedding net__: first, generate cluttered mnist dataset by running `clutter_mnist_scale.py` directly, which will
generate and save `npy` files for data and label. Then, below is an example of pretrain embedding network using digit 4 images. One could
also directly run `./pretrain.sh`. They are the same.
```bash
CUDA_VISIBLE_DEVICES=0 python ordinal-pretrain-cluttermnist-scale.py \
                       --savename cluttermnist_pretrain \
                       --digit 4 \
                       --batch_size 192 \
                       --epochs 50 \
                       --lr 0.0019125313967198946 \
                       --margin 60 \
                       --pooling_mode align \
                       --pooling_size 5 \
                       --optimizer SGD \
                       --patience 50 \
                       --step 25
```
- __Jointly train__: run `./joint.sh`.
```bash
CUDA_VISIBLE_DEVICES=0 python train-joint.py \
                       --savename cluttermnist_joint \
                       --pretrained cluttermnist_pretrain/best.pth.tar \
                       --lr_ag 0.001 \
                       --lr 0.001 \
                       --pooling_mode align \
                       --pooling_size 5 \
                       --lamb_ent 5. \
                       --seq_len 10 \
                       --margin 60 \
                       --sign 0 \
                       --hidden_size 48 \
                       --freeze 0 \
                       --norm 0 \
                       --num_act 10 \
                       --optimizer_ag Adam \
                       --optimizer SGD \
                       --epochs 250 \
                       --batch_size 512 \
                       --step_ag 80 \
                       --steps 30 70 110
```
                
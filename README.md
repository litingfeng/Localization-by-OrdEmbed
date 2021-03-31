# Localization-by-Ordinal-Embedding
This repo documents experiments steps and scripts used to localize objects with
ordinal embedding. 

## Updates
- __3/28/2021__: Add experiments for fixed size cluttered mnsit
- __3/29/2021__: Add experiments for random size & aspect ratio cluttered mnsit

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
CUDA_VISIBLE_DEVICES=0 python ordinal-pretrain-cluttermnist.py \
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
- __Visualize actions__: set path for dataloader, pretrained model and jointly trained models in 
`visualize_actions_scale.ipynb` and run.

## Random Size & Aspect Ratio Cluttered MNIST
### Description
In this section, the digit in each image become random size, from 28 to 28\*2.5=70. 
And the range of aspect ratio for the train set is [0.35, 3.1]. See the below image 
as an example.

<img src="https://github.com/litingfeng/Localization-by-Ordinal-Embedding/blob/main/images/example2.png" width="800" height="400">

- __Learn ordinal embedding__: different from previous setting, to learn more efficiently, we use dense anchor
sampling strategy. First generate dense anchors for each image. Then compute the IoUs with 
groundtruth box and divide to 10 groups. For each batch, sample two IoUs, then sample anchor boxes from the 
corresponding group. Add one more conv layer by uncommenting [this line](https://github.com/litingfeng/Localization-by-Ordinal-Embedding/blob/beddf3d64134e7ce2e7f73bdec5bce4ce4d2aa8f/models/mnist_scale_model.py#L28)
.To train, generate dataset by running `clutter_mnist_scale_anchor.py`, then replace `ordinal-pretrain-cluttermnist.py` with `ordinal-pretrain-cluttermnist-scale.py`, and
run `pretrain.sh`.
- __Jointly train__: We have 4 more actions in this setting: wider, narrower, shorter, higher. 
To train, uncomment [this part](https://github.com/litingfeng/Localization-by-Ordinal-Embedding/blob/3d2962dca71519a02476e14122956dc71b9de774/datasets/clutter_mnist_scale_rl.py#L44), 
and, change `num_act` to 14, and run `joint.sh`.  


                
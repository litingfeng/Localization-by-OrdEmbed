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
embedding space is by L2 distance. To learn this, we use embedding net shown below,  
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
                
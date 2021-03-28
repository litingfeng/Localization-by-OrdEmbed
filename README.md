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

###
                
#  WANDB_MODE=dryrun
CUDA_VISIBLE_DEVICES=0 python ordinal-pretrain-cluttermnist-scale.py \
                       --savename cluttermnist_scale_pretrain \
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

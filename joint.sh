# WANDB_MODE=dryrun
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
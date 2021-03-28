# WANDB_MODE=dryrun
CUDA_VISIBLE_DEVICES=2 python self-paced-RL-coloc-scale.py \
                       --savename  coloc_rl_seq10_scale_ent5_2fc128_AR \
                       --pretrained cluttermnistAR_m60_trip_SGD_scale_newsmp_2fc128/best.pth.tar \
                       --lr 0.001 \
                       --sparse 0 \
                       --pooling_mode align \
                       --pooling_size 5 \
                       --lamb_ent 5. \
                       --seq_len 10 \
                       --margin 60 \
                       --sign 0 \
                       --hidden_size 48 \
                       --freeze 1 \
                       --norm 0 \
                       --num_act 10 \
                       --optimizer Adam \
                       --epochs 150 \
                       --batch_size 512 \
                       --step 120

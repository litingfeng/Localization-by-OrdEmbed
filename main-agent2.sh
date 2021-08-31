# WANDB_MODE=dryrun
CUDA_VISIBLE_DEVICES=2 python baseline-pg.py \
                     --savename ae_iou_pg_ent05_lr1e03_50_noaug \
                     --pretrained ae_50_lr1e03_step50_2/last.pth.tar \
                     --sample_size 50 \
                     --evaluate 0 \
                     --seq_len 10 \
                     --digit 4 \
                     --rnn 1 \
                     --lamb_ent 0.5 \
                     --lr_ag 0.001 \
                     --hidden_size 48 \
                     --pooling_mode align \
                     --pooling_size 7 \
                     --patience 100 \
                     --freeze 1 \
                     --num_act 10 \
                     --optimizer_ag Adam \
                     --epochs 250 \
                     --batch_size 64 \
                     --step_ag 80


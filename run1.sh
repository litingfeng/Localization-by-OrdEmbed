CUDA_VISIBLE_DEVICES=0 python finetune-cub-standard.py \
                       --savename finetune_cub_ent05_warbler15_supp100_oriole_freezeall1024 \
                       --pretrained_agent ordinal_vgg_cub_seq10_ent05_lr1e03_stp80_shufprotshufprot_min40_warb15_dim1024freezeall/best.pth.tar \
                       --pretrained pretrain_vgg_ord_cub_fewshot_lr1e03_spc5_cps5_img224_warbler_shufproto_warb15_dim1024_freezeall/last.pth.tar \
                       --evaluate 0 \
                       --bg_name oriole \
                       --dim 1024 \
                       --img_size 224 \
                       --support_size 100 \
                       --min_box_side 40 \
                       --rnn 1 \
                       --lr_ag 0.001 \
                       --sparse 0 \
                       --pooling_mode align \
                       --pooling_size 7 \
                       --lamb_ent 0.5 \
                       --seq_len 10 \
                       --sign 0 \
                       --hidden_size 48 \
                       --freeze 1 \
                       --num_act 14 \
                       --patience 100 \
                       --optimizer_ag Adam \
                       --log_interval 10 \
                       --epochs 150 \
                       --batch_size 64 \
                       --step_ag 80



# WANDB_MODE=dryrun
CUDA_VISIBLE_DEVICES=3 python finetune-cub-standard.py \
                       --savename finetune_cub_ent05_warbler15_supp5_warbler_freezeall1024_supp2 \
                       --pretrained_agent ordinal_vgg_cub_seq10_ent05_lr1e03_stp80_shufprotshufprot_min40_warb15_dim1024freezeall/best.pth.tar \
                       --pretrained pretrain_vgg_ord_cub_fewshot_lr1e03_spc5_cps5_img224_warbler_shufproto_warb15_dim1024_freezeall/last.pth.tar \
                       --evaluate 0 \
                       --bg_name warbler \
                       --dim 1024 \
                       --img_size 224 \
                       --support_size 2 \
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
                       --step_ag 80 \

#CUDA_VISIBLE_DEVICES=5 python train-newdigit-2digit.py \
#                       --savename ordinal_ae_fewshot_digit34_supp5_seq10_ent6pol1_2digitpatch_trainpatch_500_shufprotoemb_2ord1m10_2m320Dim1024HR_shufprotoRew_tar3_scratch \
#                       --pretrained pretrain_ae_ord_proj_scp5_digit34_randpatch2d_shufproto_constrast_samp500_2ord_1m10_2m320_dim1024_hardneg/last.pth.tar \
#                       --evaluate 0 \
#                       --switch 0 \
#                       --dim 1024 \
#                       --bg_name random_patch_2digit_34 \
#                       --support_size 5 \
#                       --sample_size whole \
#                       --digit 3 \
#                       --rnn 1 \
#                       --lr_ag 0.001 \
#                       --pooling_mode align \
#                       --pooling_size 7 \
#                       --lamb_ent 6. \
#                       --lamb_pol 1. \
#                       --seq_len 10 \
#                       --hidden_size 48 \
#                       --freeze 1 \
#                       --norm 0 \
#                       --num_act 10 \
#                       --patience 50 \
#                       --optimizer_ag Adam \
#                       --epochs 150 \
#                       --batch_size 512 \
#                       --step_ag 80

#CUDA_VISIBLE_DEVICES=4 python train-newdigit-onlysupp.py \
#                       --savename ordinal_ae_fewshot_digit8_supp5_ent05_self_proto_onlysupp \
#                       --pretrained_agent ordinal_ae_proj_fewshot_seq10_ent6_lr1e03_stp80_50_selfsupp_proto/best.pth.tar \
#                       --pretrained pretrain_ae_ord_project_fewshot_50_lr1e03_scp5_verify/last.pth.tar \
#                       --evaluate 0 \
#                       --bg_name moreclutter \
#                       --support_size 5 \
#                       --sample_size whole \
#                       --digit 8 \
#                       --rnn 1 \
#                       --lr_ag 0.001 \
#                       --pooling_mode align \
#                       --pooling_size 7 \
#                       --lamb_ent 0.5 \
#                       --lamb_int 0. \
#                       --seq_len 10 \
#                       --margin 60 \
#                       --hidden_size 48 \
#                       --freeze 1 \
#                       --norm 0 \
#                       --num_act 10 \
#                       --patience 50 \
#                       --optimizer_ag Adam \
#                       --epochs 100 \
#                       --batch_size 512 \
#                       --step_ag 80

#CUDA_VISIBLE_DEVICES=4 python train-newdigit.py \
#                       --savename ordinal_ae_fewshot_digit2_supp2_ent05_trainpatch_shufprotoemb_shufprotoRew_ftclutter \
#                       --pretrained_agent ordinal_ae_proj_seq10_ent6_lr1e03_stp80_50_digit3_randpatch_shufprotoemb_shufprotoRew_supp5/best.pth.tar \
#                       --pretrained pretrain_ae_ord_proj_50_lr1e03_scp5_digit3_lamb01_randpatch_stp80_shuffle_proto/last.pth.tar \
#                       --evaluate 0 \
#                       --support_size 2 \
#                       --bg_name moreclutter \
#                       --sample_size whole \
#                       --digit 2 \
#                       --rnn 1 \
#                       --lr_ag 0.001 \
#                       --pooling_mode align \
#                       --pooling_size 7 \
#                       --lamb_ent 0.5 \
#                       --lamb_int 0. \
#                       --seq_len 10 \
#                       --margin 60 \
#                       --hidden_size 48 \
#                       --freeze 1 \
#                       --norm 0 \
#                       --num_act 10 \
#                       --patience 50 \
#                       --optimizer_ag Adam \
#                       --epochs 100 \
#                       --batch_size 512 \
#                       --step_ag 80
#

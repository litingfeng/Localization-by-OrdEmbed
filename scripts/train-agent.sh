#CUDA_VISIBLE_DEVICES=6 python train_agent.py \
#                       --savename github_ordinal_ae_proj_seq10_ent05_lr1e03_stp80_50_digit3_randpatch_shufprotoemb_shufprotoRew_supp10_run2 \
#                       --pretrained github_pretrain_ae_ord_proj_50_lr1e03_scp10_digit3_lamb01_randpatch_stp80_shuffle_proto_fix_run2/best.pth.tar \
#                       --img_size 84 \
#                       --bg_name patch \
#                       --samples_per_class 5 \
#                       --sample_size 50 \
#                       --digit 3 \
#                       --seq_len 10 \
#                       --num_act 10 \
#                       --patience 100 \
#                       --epochs 250 \
#                       --batch_size 50 \
#                       --steps_ag 200

#CUDA_VISIBLE_DEVICES=7 python train_agent.py \
#                       --savename github_ordinal_vgg_cub_seq10_ent05_lr1e03_stp80_shufprotshufprot_min40_warb15_dim1024freezeall \
#                       --pretrained github_pretrain_vgg_ord_cub_fewshot_lr1e03_spc25_cps5_img224_warbler_shufproto_warb15_dim1024_freezeall/last.pth.tar \
#                       --dataset cub \
#                       --dim 1024 \
#                       --dim_ag 512 \
#                       --bg_name warbler \
#                       --backbone vgg16 \
#                       --img_size 224 \
#                       --min_box_side 40 \
#                       --samples_per_class 5 \
#                       --seq_len 10 \
#                       --num_act 14 \
#                       --patience 100 \
#                       --log_interval 100 \
#                       --epochs 150 \
#                       --batch_size 50 \
#                       --steps_ag 80

CUDA_VISIBLE_DEVICES=7 python train_agent.py \
                       --savename github_ordinal_vgg_coco_seq10_ent05_lr1e03_stp80_shufprotshufprot_min40_dim1024freezeall_dog \
                       --pretrained github_pretrain_vgg_ord_coco_lr1e03_spc5_img224_dim1024_freezeall_marg60_dog_shufproto_stp80/last.pth.tar \
                       --dataset coco \
                       --backbone vgg16 \
                       --dim 1024 \
                       --dim_ag 512 \
                       --sel_cls dog \
                       --img_size 224 \
                       --min_box_side 40 \
                       --samples_per_class 5 \
                       --seq_len 10 \
                       --num_act 14 \
                       --patience 100 \
                       --log_interval 100 \
                       --epochs 150 \
                       --batch_size 50 \
                       --steps_ag 80

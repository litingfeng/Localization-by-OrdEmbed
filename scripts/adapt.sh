#CUDA_VISIBLE_DEVICES=6 python adapt_agent.py \
#                       --savename github_ordinal_ae_fewshot_digit2_supp5_ent05_trainpatch_shufprotoemb_shufprotoRew_ftclutter_run2 \
#                       --pretrained_agent github_ordinal_ae_proj_seq10_ent05_lr1e03_stp80_50_digit3_randpatch_shufprotoemb_shufprotoRew_supp10_run2/best.pth.tar \
#                       --pretrained github_pretrain_ae_ord_proj_50_lr1e03_scp10_digit3_lamb01_randpatch_stp80_shuffle_proto_fix_run2/last.pth.tar \
#                       --support_size 5 \
#                       --bg_name clutter \
#                       --sample_size whole \
#                       --digit 2 \
#                       --seq_len 10 \
#                       --num_act 10 \
#                       --patience 50 \
#                       --epochs 100 \
#                       --batch_size 512 \
#                       --steps_ag 80


#CUDA_VISIBLE_DEVICES=7 python adapt_agent.py \
#                       --savename github_finetune_cub_ent05_supp5_warbler2gull_freezeall1024 \
#                       --pretrained_agent github_ordinal_vgg_cub_seq10_ent05_lr1e03_stp80_shufprotshufprot_min40_warb15_dim1024freezeall/best.pth.tar \
#                       --pretrained github_pretrain_vgg_ord_cub_fewshot_lr1e03_spc25_cps5_img224_warbler_shufproto_warb15_dim1024_freezeall/last.pth.tar \
#                       --evaluate \
#                       --bg_name gull \
#                       --dataset cub \
#                       --backbone vgg16 \
#                       --dim 1024 \
#                       --dim_ag 512 \
#                       --img_size 224 \
#                       --support_size 5 \
#                       --min_box_side 40 \
#                       --seq_len 10 \
#                       --num_act 14 \
#                       --patience 100 \
#                       --log_interval 10 \
#                       --epochs 150 \
#                       --batch_size 64 \
#                       --steps_ag 80

CUDA_VISIBLE_DEVICES=6 python adapt_agent.py \
                       --savename github_finetune_coco_ent05_supp5_freezeall1024_cow2cat \
                       --pretrained_agent github_ordinal_vgg_coco_seq10_ent05_lr1e03_stp80_shufprotshufprot_min40_dim1024freezeall_cow/best.pth.tar \
                       --pretrained github_pretrain_vgg_ord_coco_lr1e03_spc5_img224_dim1024_freezeall_marg60_cow_shufproto_stp80/last.pth.tar \
                        --dataset coco \
                       --backbone vgg16 \
                       --sel_cls cat \
                       --dim 1024 \
                       --dim_ag 512 \
                       --img_size 224 \
                       --support_size 5 \
                       --min_box_side 40 \
                       --seq_len 10 \
                       --num_act 14 \
                       --patience 100 \
                       --log_interval 10 \
                       --epochs 150 \
                       --batch_size 64 \
                       --steps_ag 80
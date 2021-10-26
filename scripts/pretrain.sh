#CUDA_VISIBLE_DEVICES=7 python pretrain_encoder_ordinal.py \
#                         --savename github_pretrain_ae_ord_proj_50_lr1e03_scp10_digit3_lamb01_randpatch_stp80_shuffle_proto_fix_run2 \
#                         --digit 3 \
#                         --bg_name patch \
#                         --lamb 0.1 \
#                         --margin 60 \
#                         --samples_per_class 10 \
#                         --sample_size 50 \
#                         --batch_size 50 \
#                         --patience 150 \
#                         --epochs 150 \
#                         --steps 80


# pretrain on cub dataset
CUDA_VISIBLE_DEVICES=6 python pretrain_encoder_ordinal.py \
                        --savename github_pretrain_vgg_ord_cub_spc5_cps5_warbler_warb15 \
                        --dataset cub \
                        --backbone vgg16 \
                        --bg_name warbler \
                        --dim 1024 \
                        --batch_size 50 \
                        --img_size 224 \
                        --log_interval 25 \
                        --samples_per_class 5 \
                        --lamb 1.0 \
                        --epochs 150 \
                        --margin 60 \
                        --patience 120 \
                        --steps 80

# pretrain on coco dataset
#CUDA_VISIBLE_DEVICES=7 python pretrain_encoder_ordinal.py \
#                        --savename github_pretrain_vgg_ord_coco_lr1e03_spc5_img224_dim1024_freezeall_marg60_dog_shufproto_stp80 \
#                        --dataset coco \
#                        --sel_cls dog \
#                        --backbone vgg16 \
#                        --dim 1024 \
#                        --batch_size 75 \
#                        --img_size 224 \
#                        --log_interval 25 \
#                        --samples_per_class 5 \
#                        --lamb 1.0 \
#                        --epochs 200 \
#                        --margin 60 \
#                        --patience 120 \
#                        --steps 80
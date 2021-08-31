#  WANDB_MODE=dryrun
#CUDA_VISIBLE_DEVICES=2 python omg_ordinal_fewshot.py \
#                        --savename pretrain_ae_ord_omg_fewshot_5n5k_25cls_iter50_self \
#                        --evaluate 0 \
#                        --bg_name patch \
#                        --train_mode self \
#                        --img_size 84 \
#                        --iterations 50 \
#                        --iterations_test 50 \
#                        --samples_per_class 5 \
#                        --classes_per_set 5 \
#                        --log_interval 50 \
#                        --lamb 0.1 \
#                        --margin 60 \
#                        --pooling_mode align \
#                        --pooling_size 7 \
#                        --patience 120 \
#                        --epochs 150 \
#                        --optimizer Adam \
#                        --lr 0.001 \
#                        --steps 50
#CUDA_VISIBLE_DEVICES=0 python omg_autoencoder_ordinal_fewshot_siamese2.py \
#                        --savename omg_pretrain_ae_ord_fewshot_25cls_5n5k_2digit_lambcls05_onlycls \
#                        --evaluate 0 \
#                        --iterations 100 \
#                        --samples_per_class 5 \
#                        --classes_per_set 5 \
#                        --log_interval 50 \
#                        --lamb 0.1 \
#                        --lamb_cls 1. \
#                        --margin 60 \
#                        --margin2 60 \
#                        --pooling_mode align \
#                        --pooling_size 7 \
#                        --patience 120 \
#                        --epochs 150 \
#                        --optimizer Adam \
#                        --lr 0.001 \
#                        --steps 50
#CUDA_VISIBLE_DEVICES=0 python mnist_fixedEmb_ordinal_2digit_2pair.py \
#                        --savename pretrain_siam_ae_ord_scp5_digit34_2d_500_2ord_mar10_dim1024_lamb05_longersame \
#                        --pretrained_cls mnist_siamese_mar320_500_dim1024_longer_same/last.pth.tar \
#                        --evaluate 0 \
#                        --dim 1024 \
#                        --bg_name random_patch_2digit_34 \
#                        --digit 3 \
#                        --samples_per_class 5 \
#                        --lamb 0.5 \
#                        --margin 10 \
#                        --sample_size 500 \
#                        --batch_size 50 \
#                        --pooling_mode align \
#                        --pooling_size 7 \
#                        --patience 150 \
#                        --epochs 250 \
#                        --optimizer Adam \
#                        --lr 0.001 \
#                        --steps 80
CUDA_VISIBLE_DEVICES=0 python mnist_autoencoder_ordinal_fewshot.py \
                        --savename pretrain_ae_ord_proj_50_lr1e03_scp5_digit3_lamb01_randpatch_stp80_shuffle_proto_every  \
                        --evaluate 0 \
                        --digit 3 \
                        --lamb 0.1 \
                        --bg_name random_patch \
                        --train_mode shuffle_proto \
                        --sample_size 50 \
                        --batch_size 500 \
                        --samples_per_class 5 \
                        --pooling_mode align \
                        --pooling_size 7 \
                        --margin 60 \
                        --patience 150 \
                        --epochs 150 \
                        --optimizer Adam \
                        --lr 0.001 \
                        --steps 50
#CUDA_VISIBLE_DEVICES=0 python mnist_autoencoder_ordinal.py \
#                         --savename pretrain_ae_ord_proj_50_lr1e03_scp5_digit3_lamb01_randpatch_stp80_self \
#                         --evaluate 0 \
#                         --digit 3 \
#                         --bg_name random_patch \
#                         --lamb 0.1 \
#                         --margin 60 \
#                         --sample_size 50 \
#                         --batch_size 50 \
#                         --pooling_mode align \
#                         --pooling_size 7 \
#                         --patience 150 \
#                         --epochs 150 \
#                         --optimizer Adam \
#                         --lr 0.001 \
#                         --steps 80

#CUDA_VISIBLE_DEVICES=5 python mnist_autoencoder.py \
#                        --savename ae_50_lr1e03_step50_digit3_randpatch \
#                        --bg_name random_patch \
#                        --evaluate 0 \
#                        --digit 3 \
#                        --sample_size 50 \
#                        --batch_size 32 \
#                        --epochs 150 \
#                        --optimizer Adam \
#                        --lr 0.001 \
#                        --steps 50
#CUDA_VISIBLE_DEVICES=5 python cub_ordinal_fewshot.py \
#                        --savename pretrain_vgg_ord_cub_fewshot_5n5k_iter100_lambcls1_proto_entcls_frzall_1024_r2 \
#                        --evaluate 0 \
#                        --net vgg16 \
#                        --dim 1024 \
#                        --num_cls 15 \
#                        --freeze 1 \
#                        --img_size 224 \
#                        --train_mode proto \
#                        --iterations 100 \
#                        --iterations_test 50 \
#                        --log_interval 25 \
#                        --samples_per_class 5 \
#                        --classes_per_set 5 \
#                        --lamb 1. \
#                        --lamb_cls 1. \
#                        --epochs 200 \
#                        --optimizer Adam \
#                        --margin 60 \
#                        --patience 120 \
#                        --pooling_mode align \
#                        --pooling_size 7 \
#                        --lr 0.001 \
#                        --steps 80
#CUDA_VISIBLE_DEVICES=3 python cub_pretrain_ordinal_fewshot.py \
#                        --savename pretrain_vgg_ord_cub_fewshot_lr1e03_spc25_cps5_img224_warbler_shufproto_warb15_dim1024_freezeall_r2 \
#                        --evaluate 0 \
#                        --loose 0 \
#                        --net vgg16 \
#                        --dim 1024 \
#                        --batch_size 50 \
#                        --freeze 1 \
#                        --img_size 224 \
#                        --train_mode shuffle_proto \
#                        --iterations 50 \
#                        --iterations_test 100 \
#                        --log_interval 25 \
#                        --samples_per_class 25 \
#                        --classes_per_set 5 \
#                        --lamb 1.0 \
#                        --lamb_cls 0. \
#                        --epochs 150 \
#                        --optimizer Adam \
#                        --margin 60 \
#                        --margin_cls 10 \
#                        --patience 120 \
#                        --pooling_mode align \
#                        --pooling_size 7 \
#                        --lr 0.001 \
#                        --steps 80
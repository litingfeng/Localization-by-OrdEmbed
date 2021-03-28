#  WANDB_MODE=dryrun
CUDA_VISIBLE_DEVICES=0 python ordinal-loss-newtrip-cluttermnist-scale-ar.py \
                       --savename cluttermnistARLarger_m60_trip_SGD_scale_anchorsmp_2fc5conv \
                       --digit 4 \
                       --batch_size 192 \
                       --evaluate 0 \
                       --epochs  50 \
                       --lr 0.0019125313967198946 \
                       --margin 60 \
                       --pooling_mode align \
                       --pooling_size 5 \
                       --optimizer SGD \
                       --patience 50  \
                       --step 25
#CUDA_VISIBLE_DEVICES=0 python ordinal-feat.py \
#                       --savename ordinal_feat_triplet_st80_m10_lm1 \
#                       --lr 0.001 \
#                       --margin 10.0 \
#                       --lamb 1. \
#                       --norm 0 \
#                       --optimizer Adam \
#                       --epochs 200 \
#                       --batch_size 512 \
#                       --step 80
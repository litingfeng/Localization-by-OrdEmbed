CUDA_VISIBLE_DEVICES=0 python ordinal-regression.py \
                       --savename order_reg_cumloss \
                       --lr 0.001 \
                       --angle_step 15 \
                       --mode cum_loss \
                       --optimizer Adam \
                       --epochs 200 \
                       --batch_size 512 \
                       --step 80
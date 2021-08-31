# WANDB_MODE=dryrun
CUDA_VISIBLE_DEVICES=2 WANDB_MODE=dryrun python baseline-DQN-batch-ae.py \
                       --savename  baseline_dqn_seq10_eps5epoch_uptar1_h48_bf6e4_mse_stp25_batch_ae_Conv32_lr1e04_dp075 \
                       --evaluate 1 \
                       --seq_len 10 \
                       --hist 0 \
                       --eps_decay_steps 5 \
                       --eps_start 1.0 \
                       --update_target_every 1 \
                       --digit 0 \
                       --lr_ag 0.0001 \
                       --hidden_size 48 \
                       --buffer_size 60000 \
                       --pooling_mode align \
                       --pooling_size 7 \
                       --freeze 1 \
                       --num_act 10 \
                       --optimizer_ag Adam \
                       --epochs 50 \
                       --batch_size 512 \
                       --step_ag 25 \

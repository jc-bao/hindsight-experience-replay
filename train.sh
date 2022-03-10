tmux rename-window 1obj_inhand_attn4_seed300
export OMPI_MCA_btl_vader_single_copy_mechanism=none
mpirun --allow-run-as-root -np 64 python train.py \
--n-epochs 100 --env-name PandaTowerBimanualInHand-v1 --max-trail-time 10 --curriculum --curriculum-init 0.5 --curriculum-bar 0.8 --curriculum-step 0.02 --wandb --project Bimanual --group 1obj --name 1obj_inhand_attn4_seed300 --use-attn --num-blocks 4 --seed 300 --extra-reset-steps --eval-kwargs "{\"num_need_handover\": [0, 1]}" --resume
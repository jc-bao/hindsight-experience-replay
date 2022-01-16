tmux rename-window formation_central
export OMPI_MCA_btl_vader_single_copy_mechanism=none
mpirun --allow-run-as-root -np 64 python train.py \
--n-epoch 200 \
--env-name formation_hd_env --num-agents 32 --dim 2 \
--actor-large \
--wandb --project Formation --name hd32_central
tmux rename-window hd32_central
export OMPI_MCA_btl_vader_single_copy_mechanism=none
mpirun --allow-run-as-root -np 32 python train.py \
--env-name formation_hd_env --num-agents 8 --dim 2 \
--actor-shared \
--wandb --project formation --name hd8_seperated
tmux rename-window hd32_shared_id
export OMPI_MCA_btl_vader_single_copy_mechanism=none
mpirun --allow-run-as-root -np 64 python train.py \
--env-name formation_hd_env --num-agents 32 --dim 2 \
--n-epoch 200 \
--actor-shared \
--wandb --project formation --name hd32_shared
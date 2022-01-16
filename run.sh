tmux rename-window formation_central
export OMPI_MCA_btl_vader_single_copy_mechanism=none
mpirun --allow-run-as-root -np 2 python train.py \
--env-name formation_hd_env --num-agents 64 --dim 2 \
--actor-large \
# --wandb --project Formation --name hd64_central 
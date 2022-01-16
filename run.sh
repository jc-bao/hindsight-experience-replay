tmux rename-window hd51
export OMPI_MCA_btl_vader_single_copy_mechanism=none
mpirun --allow-run-as-root -np 32 \
python train.py \
--env-name 'formation_hd_env' \
--num-agents 4 --dim 2 \
--replay-k 0 \
--seed 1 \
--actor-shared \
--reward-type 'hd' \
--wandb --project MPE --name hd5 --group hd5
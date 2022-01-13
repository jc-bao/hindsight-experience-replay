mpirun -np 16 \
python ../../train.py \
--env-name 'formation_hd_env' \
--num-agents 4 --dim 2 \ # ma settings
--actor-separated \
--wandb --project Formation --name hd4_separated
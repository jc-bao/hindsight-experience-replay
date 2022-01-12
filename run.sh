mpirun --allow-run-as-root -np 16 python train.py \
--multi-agent --num-agents 5 --dim 2 \
--env-name formation_hd_env \
--max-trail-time 10 --trail-mode any \
--wandb --project Formation \
--name hd5_ma
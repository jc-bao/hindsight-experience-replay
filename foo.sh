mpirun --allow-run-as-root -np 16 python train.py \
--env-name PandaRearrange-v2 --max-trail-time 10 \
--not-relabel-unmoved --random-unmoved \
--wandb --project Relabel --group 1arm \
--name moved_random \
--render
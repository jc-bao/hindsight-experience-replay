mpirun --allow-run-as-root -np 64 python train.py --n-epochs 200 \
--env-name PandaTowerBimanual-v2 --max-trail-time 10 \
--not-relabel-unmoved --random-unmoved \
--wandb --project Bimanual --group 2obj \
--name 2obj_moved_and_random --note fix_in_air_and_fix_her \
--render 
mpirun --allow-run-as-root -np 64 python train.py --n-epochs 200 \
--env-name PandaTowerBimanual-v2 \
--curriculum --curriculum-bar 0.9 \
--max-trail-time 10 --use-critic-sum \
--wandb --project Bimanual --group 2obj \
--name 2obj_critic_sum_curri --note use_critic_sum_curri \
--render 
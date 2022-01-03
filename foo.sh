mpirun --allow-run-as-root -np 64 python train.py --n-epochs 200 \
--env-name PandaTowerBimanual-v2 \
--curriculum --curriculum-reward --curriculum-bar -1.1 \
--max-trail-time 10 --use-critic-sum \
--wandb --project Bimanual --group 2obj \
--name 2obj_critic_sum_fix_curriculum --note use_critic_sum_curri \
--render --resume
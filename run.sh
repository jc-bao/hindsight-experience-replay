mpirun --allow-run-as-root -np 16 \
python train.py \
--env-name 'formation_hd_env' \
--num-agents 4 --dim 2 \
--actor-shared \
--n-batches 100 \
--not-resume-actor \
--resume --model-path '/rl/hindsight-experience-replay/saved_models/model.pt' \
--lr-critic 0.0001 \
--wandb --project Formation --name hd4_resume_central_batch10_critic_lr 
# --wandb --project Formation --name hd4_n_batch10 \
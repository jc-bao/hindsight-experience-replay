mpirun --allow-run-as-root -np 32 \
python train.py \
--env-name 'formation_hd_env' \
--num-agents 4 --dim 2 \
--wandb --project Formation --name hd4_actorup10_shared_resume
--actor-update-times 10 \
--actor-shared --not-resume-actor \
--resume --model-path '/rl/hindsight-experience-replay/saved_models/formation_hd_env/hd4_central/model.pt' \
# --wandb --project Formation --name hd8_central_new
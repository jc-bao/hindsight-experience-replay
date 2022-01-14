mpirun --allow-run-as-root -np 32 \
python train.py --env-name formation_hd_env --num-agents 16 --dim 2 --wandb --project Formation --name hd16_central \
--resume --model-path '/rl/hindsight-experience-replay/saved_models/formation_hd_env/hd8_central/model.pt'
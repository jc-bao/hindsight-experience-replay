tmux rename-window hd8_dropout0.5_resume
export OMPI_MCA_btl_vader_single_copy_mechanism=none
mpirun --allow-run-as-root -np 32 \
python train.py \
--env-name 'formation_hd_env' \
--num-agents 8 --dim 2 \
--actor-dropout --drop-out-rate 0.5 \
--wandb --project formation --name hd8_dropout0.5_resume \
--resume --model-path '/rl/hindsight-experience-replay/saved_models/formation_hd_env/hd8_central.pt'
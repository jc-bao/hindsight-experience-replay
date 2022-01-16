tmux rename-window hd4_dropout1_resume
export OMPI_MCA_btl_vader_single_copy_mechanism=none
mpirun --allow-run-as-root -np 16 \
python train.py \
--env-name 'formation_hd_env' \
--num-agents 4 --dim 2 \
--actor-dropout \
--wandb --project formation --name hd4_dropout1_resume \
--resume --model-path '/rl/hindsight-experience-replay/saved_models/formation_hd_env/hd4_dropout_resume/model.pt'
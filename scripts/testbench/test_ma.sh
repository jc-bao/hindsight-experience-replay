python ../../train.py \
--n-epochs 2 --n-cycles 2 --n-batches 1 --buffer-size 1000 --batch-size 32 \
--actor-multihead \
--num-agents 4 --dim 2 \
--env-name 'formation_hd_env' \
# --resume --model-path '/rl/hindsight-experience-replay/saved_models/formation_hd_env/hd4_dropout_resume/model.pt'
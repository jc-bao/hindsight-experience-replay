python -m cProfile -o formation.prof ../../train.py \
--n-epochs 2 --n-cycles 20 --n-batches 10 --buffer-size 10000 --batch-size 3200 \
--actor-shared \
--num-agents 4 --dim 2 \
--env-name 'formation_hd_env'
# --resume --model-path '/rl/hindsight-experience-replay/saved_models/formation_hd_env/hd4_dropout_resume/model.pt'
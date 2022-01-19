python ../train.py \
--n-epochs 10 --n-cycles 0 --n-batches 0 --buffer-size 1000 --batch-size 32 \
--env-name 'formation_hd_env' --num-agents 16 --dim 2 --actor-shared \
--resume --model-path '/Users/reedpan/Downloads/model.pt' --render
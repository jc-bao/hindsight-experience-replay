python ../../train.py \
--n-epochs 2 --n-cycles 2 --n-batches 1 --buffer-size 1000 --batch-size 32 \
--actor-dropout \
--num-agents 4 --dim 2 \
--env-name 'formation_hd_env' \
--resume --model-path '/Users/reedpan/Downloads/model.pt'
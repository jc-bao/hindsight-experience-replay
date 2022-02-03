python ../../train.py \
--n-epochs 10 --n-cycles 2 --n-batches 1 --buffer-size 1000 --batch-size 32 \
--env-name 'PandaRearrange-v1' \
--curriculum --curriculum-init 1 --curriculum-end 6 --curriculum-step 1 --curriculum-bar '-0.1' \
--use-attn
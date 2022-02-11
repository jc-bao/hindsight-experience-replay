python ../train.py \
--n-epochs 10 --n-cycles 0 --n-batches 0 --buffer-size 1000 --batch-size 32 \
--env-name PandaTowerBimanualMaxHandover1SlowNoise-v2 --use-attn --num-blocks 4 \
--gui --resume --model-path '/Users/reedpan/Downloads/hand2_attn4.pt'
# --curriculum --curriculum-init 1 --curriculum-end 1 --curriculum-step 0.1 --curriculum-bar '-0.1' --curriculum-type 'dropout' \
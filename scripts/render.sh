python ../train.py \
--n-epochs 2 --n-cycles 1 --n-batches 0 --buffer-size 1000 --batch-size 32 \
--env-name PandaTowerBimanualMaxHandover1-v4 --use-attn --num-blocks 4 \
--gui --resume --model-path '/Users/reedpan/Downloads/1hand1re.pt' \
--curriculum --curriculum-init 4 --curriculum-end 4.5 --curriculum-step 0.1 --curriculum-bar '-0.1' --n-test-rollouts 1
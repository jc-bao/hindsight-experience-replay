python ../train.py \
--n-epochs 10 --n-cycles 0 --n-batches 0 --buffer-size 1000 --batch-size 32 --n-test-rollouts 2 \
--env-name PandaTowerBimanualParallelSlow-v2 \
--resume --model-path '/Users/reedpan/Downloads/handover2_os08.pt' \
--store-trajectory --store-video \
--use-attn --num-blocks 4 \
# --gui \
# --curriculum --curriculum-init 1.4 --curriculum-end 2 --curriculum-bar '-0.1' --curriculum-step 0.1 \
python ../train.py \
--n-epochs 10 --n-cycles 0 --n-batches 0 --buffer-size 1000 --batch-size 32 --n-test-rollouts 2 \
--env-name PandaTowerBimanual-v1 \
--resume --model-path '/Users/reedpan/Downloads/curr1.50_best_model.pt' \
--gui \
--use-attn --num-blocks 4 \
# --store-trajectory --store-video \
# --curriculum --curriculum-init 1.4 --curriculum-end 2 --curriculum-bar '-0.1' --curriculum-step 0.1 \
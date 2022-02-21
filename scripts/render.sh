python ../train.py \
--n-epochs 10 --n-cycles 0 --n-batches 0 --buffer-size 1000 --batch-size 32 --n-test-rollouts 2 \
--env-name PandaTowerBimanual-v1 \
--resume --model-path '/Users/reedpan/Downloads/handover2_os08.pt' \
--use-attn --num-blocks 4 \
--env-kwargs '{"os_rate":1, "render":true,"store_trajectory":false, "store_video":false}'
# --store-trajectory --store-video \
# --curriculum --curriculum-init 1.4 --curriculum-end 2 --curriculum-bar '-0.1' --curriculum-step 0.1 \
python ../train.py \
--n-epochs 10 --n-cycles 0 --n-batches 0 --buffer-size 1000 --batch-size 32 --n-test-rollouts 10 \
--env-name PandaTowerBimanual-v1 \
--resume --model-path '/Users/reedpan/Downloads/nobj_attn3_handnum.pt' \
--env-kwargs '{"os_rate":1, "render":true}' \
--use-attn --num-blocks 3

 # "use_bound": false, "store_trajectory":false, "store_video":false, "obj_in_hand_rate":0}' \
# --extra-reset-steps
# --store-trajectory --store-video \
# --curriculum --curriculum-init 1.4 --curriculum-end 2 --curriculum-bar '-0.1' --curriculum-step 0.1 \
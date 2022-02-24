python ../train.py \
--n-epochs 10 --n-cycles 0 --n-batches 0 --buffer-size 1000 --batch-size 32 --n-test-rollouts 10 \
--env-name PandaTowerBimanualParallel-v2 \
--resume --model-path '/Users/reedpan/Downloads/handover2_os08.pt' \
--env-kwargs '{"os_rate":1, "render":true, "obj_in_hand_rate":0, "subgoal_rate": 1, "reward_type": "final"}' \
--use-attn --num-blocks 4

 # "use_bound": false, "store_trajectory":false, "store_video":false, "obj_in_hand_rate":0}' \
# --extra-reset-steps
# --store-trajectory --store-video \
# --curriculum --curriculum-init 1.4 --curriculum-end 2 --curriculum-bar '-0.1' --curriculum-step 0.1 \
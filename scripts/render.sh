python ../train.py \
--n-epochs 10 --n-cycles 0 --n-batches 0 --buffer-size 1000 --batch-size 32 --n-test-rollouts 10 \
--env-name PandaTowerBimanual-v1 \
--resume --model-path '/Users/reedpan/Downloads/1obj_best_model.pt' \
--env-kwargs '{"os_rate":0.5, "render":true, "goal_in_obj_rate": 0, "goal_range": [0.4, 0.3, 0], "obj_in_hand_rate": 0}' \
--use-attn --num-blocks 3
# --shared-normalizer \
# --extra-reset-steps

 # "use_bound": false, "store_trajectory":false, "store_video":false, "obj_in_hand_rate":0}' \
# --extra-reset-steps
# --store-trajectory --store-video \
# --curriculum --curriculum-init 1.4 --curriculum-end 2 --curriculum-bar '-0.1' --curriculum-step 0.1 \
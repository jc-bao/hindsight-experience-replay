python ../train.py \
--n-epochs 10 --n-cycles 0 --n-batches 0 --buffer-size 1000 --batch-size 32 --n-test-rollouts 100 \
--env-name PandaTowerBimanualSlow-v4 \
--resume --model-path '/Users/reedpan/Downloads/1hand_3re_attn5.pt' \
--env-kwargs '{"os_rate":0.8, "render":false, "obj_in_hand_rate":0, "max_num_need_handover": 1, "store_trajectory":true, "store_video":true}' \
--use-attn --num-blocks 5

 # "use_bound": false, "store_trajectory":false, "store_video":false, "obj_in_hand_rate":0}' \
# --extra-reset-steps
# --store-trajectory --store-video \
# --curriculum --curriculum-init 1.4 --curriculum-end 2 --curriculum-bar '-0.1' --curriculum-step 0.1 \
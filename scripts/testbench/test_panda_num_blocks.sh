python ../../train.py \
--n-epochs 10 --n-cycles 2 --n-batches 1 --buffer-size 1000 --batch-size 32 \
--env-name 'PandaTowerBimanualInHand-v1' \
--curriculum --curriculum-init 0.5 --curriculum-end 1 --curriculum-step 0.04 \
--curriculum-bar '-0.1' --curriculum-indicator num_need_handover0.00 \
 --curriculum-indicator num_need_handover0.00 \
--use-attn --num-blocks 3 \
--eval-kwargs '{"num_need_handover": [0, 1]}' \
--env-kwargs '{"use_bound": true}' \
--random-unmoved-rate 1 --not-relabel-unmoved \
--change-os-rate

# mpirun -np 2 
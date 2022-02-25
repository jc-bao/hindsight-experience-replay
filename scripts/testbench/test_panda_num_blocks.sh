mpirun -np 2 \
python ../../train.py \
--n-epochs 10 --n-cycles 2 --n-batches 1 --buffer-size 1000 --batch-size 32 \
--env-name 'PandaTowerBimanual-v0' \
 --curriculum-indicator num_need_handover0.00 \
--use-attn --num-blocks 3 \
--eval-kwargs '{"num_need_handover": [0, 1]}' \
--env-kwargs '{"use_bound": true, "num_blocks": 2}' \
--random-unmoved-rate 1 --not-relabel-unmoved \
--change-os-rate \
--easy-env-num 1 \
--shared-normalizer \
--curriculum --curriculum-init 1 --curriculum-end 3 --curriculum-step 1 \
--curriculum-bar '-0.1' --curriculum-indicator num_need_handover0.00 \

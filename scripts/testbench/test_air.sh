python ../../train.py \
--n-epochs 10 --n-cycles 2 --n-batches 1 --buffer-size 1000 --batch-size 32 \
--env-name 'PandaTowerBimanual-v1' \
--use-air \
--replay-strategy 'none'
# --curriculum --curriculum-bar '-0.1' --curriculum-init 1.1 \
# --env-kwargs '{"use_task_distribution": true}' \
# --eval-kwargs '{"num_need_handover": [0, 1]}' \
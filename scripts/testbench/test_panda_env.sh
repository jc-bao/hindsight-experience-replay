python ../../train.py \
--n-epochs 10 --n-cycles 2 --n-batches 1 --buffer-size 1000 --batch-size 32 \
--curriculum --curriculum-bar '-0.1' --curriculum-type 'task_distribution' \
--eval-kwargs '{"num_need_handover": [0, 1]}' \
--env-name 'PandaTowerBimanualTaskDistribution-v1'
python ../../train.py \
--n-epochs 10 --n-cycles 2 --n-batches 1 --buffer-size 1000 --batch-size 32 \
--curriculum --curriculum-init 0 --curriculum-end 1 --curriculum-step 0.1 --curriculum-bar '-0.1' --curriculum-type 'dropout' \
--env-name 'PandaTowerBimanual-v1'
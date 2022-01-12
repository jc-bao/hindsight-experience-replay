python train.py \
--n-epochs 10 --n-cycles 2 --n-batches 1 --buffer-size 1000 --batch-size 32 \
--env-name 'PandaTowerBimanualNumBlocks-v2' \
--curriculum --curriculum-init 1 --curriculum-end 3 --curriculum-step 1 --curriculum-bar '-0.1' \
--use-renn
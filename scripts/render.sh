python ../train.py \
--n-epochs 20 --n-cycles 0 --n-batches 0 --buffer-size 1000 --batch-size 32 \
--env-name PandaTowerBimanualSingleSide-v1 --resume --model-path '/Users/reedpan/Desktop/Research/hindsight-experience-replay/scripts/saved_models/single_side.pt' --actor-master-slave --gui
python ../train.py \
--n-epochs 1 --n-cycles 2 --n-batches 1 --buffer-size 1000 --batch-size 32 \
--env-name 'FetchBlockConstruction_2Blocks_IncrementalReward_DictstateObs_42Rendersize_TrueStackonly_SingletowerCase-v1' \
--use-renn \
--resume \
--model-path '/Users/reedpan/Desktop/Research/hindsight-experience-replay/test/saved_models/FetchBlockConstruction_1Blocks_IncrementalReward_DictstateObs_42Rendersize_FalseStackonly_SingletowerCase-v1/noname/model.pt'
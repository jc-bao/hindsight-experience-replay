# mpirun --allow-run-as-root -np 35 \
mpirun -np 2 \
	python train.py \
	--lr-actor 0.0003 --lr-critic 0.0003 \
	--buffer-size 100000 \
	--env-name 'FetchBlockConstruction_2Blocks_IncrementalReward_DictstateObs_42Rendersize_TrueStackonly_SingletowerCase-v1' \
	--max-trail-time 10 \
	--use-renn \
	--resume --model-path '/Users/reedpan/Downloads/model.pt'
	# --wandb --project Multi-Object --group 1obj \
	# --name ReNN2-stackonly \
	# --render \
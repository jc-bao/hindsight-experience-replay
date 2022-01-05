mpirun -np 4 python train.py \
--env-name PandaRearrange-v2 --max-trail-time 10 \
--her-batch \
--n-epoch 2 --n-cycle 2 --n-batch 2 --batch-size 8 \
--not-relabel-unmoved --random-unmoved \
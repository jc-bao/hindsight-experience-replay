tmux rename-window nobj_biattn
export OMPI_MCA_btl_vader_single_copy_mechanism=none
mpirun --allow-run-as-root -np 64 python train.py \
--n-epochs 200 --env-name PandaTowerBimanualNumBlocks-v2 \
--use-biattn --max-trail-time 10 --trail-mode any \
--not-relabel-unmoved --random-unmoved \
--curriculum --curriculum-init 1 --curriculum-end 6 --curriculum-step 1 --curriculum-bar 0.9 \
--wandb --project Bimanual --name nobj_biattn --render
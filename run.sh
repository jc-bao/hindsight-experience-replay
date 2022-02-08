tmux rename-window nobj_attn_handnum
export OMPI_MCA_btl_vader_single_copy_mechanism=none
mpirun --allow-run-as-root -np 64 python train.py \
--n-epochs 200 --env-name PandaTowerBimanualHandNumMix-v1 --use-attn --num-blocks 2 --max-trail-time 10 --trail-mode any --not-relabel-unmoved --random-unmoved --curriculum --curriculum-init 1 --curriculum-end 2 --curriculum-step 0.2 --curriculum-bar 0.7 --wandb --project Bimanual --name nobj_attn_handnum --render --extra-reset-steps --resume
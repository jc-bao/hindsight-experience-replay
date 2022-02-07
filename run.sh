tmux rename-window 2obj_master_attn
export OMPI_MCA_btl_vader_single_copy_mechanism=none
mpirun --allow-run-as-root -np 35 python train.py \
--n-epochs 100 --env-name PandaTowerBimanualSingleSide-v1 --actor-master-slave --use-attn --num-blocks 2 --master-only --max-trail-time 10 --trail-mode any --not-relabel-unmoved --random-unmoved --curriculum --curriculum-init 1 --curriculum-end 2 --curriculum-step 1 --curriculum-bar 0.9 \
--wandb --project Bimanual --name 2obj_master_attn --render
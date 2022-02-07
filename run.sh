mpirun -np 1 python train.py \
--env-name PandaTowerBimanualSingleSide-v2 --actor-master-slave --use-attn --num-blocks 2 --master-only --max-trail-time 10 --trail-mode any --not-relabel-unmoved --random-unmoved --extra-reset-steps
# tmux rename-window 1obj_master_slave_resume_from_single
# export OMPI_MCA_btl_vader_single_copy_mechanism=none
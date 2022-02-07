tmux rename-window 1obj_master_slave_resume_from_single
export OMPI_MCA_btl_vader_single_copy_mechanism=none
mpirun --allow-run-as-root -np 1 python train.py \
--env-name PandaTowerBimanual-v1 --actor-master-slave --max-trail-time 10 --trail-mode any --not-relabel-unmoved --random-unmoved --wandb --project Bimanual --name 1obj_master_slave_resume_from_single --render --extra-reset-steps --resume --model-path '/rl/hindsight-experience-replay/saved_models/PandaTowerBimanualSingleSide-v1/1obj_master/model.pt'
tmux rename-window hand1_ren_mirrorattn
export OMPI_MCA_btl_vader_single_copy_mechanism=none
mpirun --allow-run-as-root -np 64 python train.py \
--n-epochs 100 --env-name PandaTowerBimanualMaxHandover1-v1 --actor-master-slave --use-attn --num-blocks 4 --shared-policy --max-trail-time 10 --trail-mode any --not-relabel-unmoved --random-unmoved --curriculum --curriculum-init 1 --curriculum-end 3 --curriculum-step 0.5 --curriculum-bar 0.9 --wandb --project Bimanual --name hand1_ren_mirrorattn --render
tmux rename-window 2obj_hand_bi
export OMPI_MCA_btl_vader_single_copy_mechanism=none
mpirun --allow-run-as-root -np 64 python train.py \
--n-epochs 200 --env-name PandaTowerBimanualInHand-v2 --max-trail-time 10 --curriculum --curriculum-init 0.3 --curriculum-bar 0.9 --wandb --project Bimanual --group 2obj --name 2obj_inhands --note 2obj_hand_bi --render --use-bilinear --extra-reset-steps
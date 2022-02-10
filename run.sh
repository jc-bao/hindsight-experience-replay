tmux rename-window hand1_rex_attn4
export OMPI_MCA_btl_vader_single_copy_mechanism=none
mpirun --allow-run-as-root -np 64 python train.py \
--n-epochs 100 --env-name PandaTowerBimanualMaxHandover1-v2 --use-attn --num-blocks 4 --max-trail-time 10 --trail-mode any --not-relabel-unmoved --random-unmoved --curriculum --curriculum-init 2 --curriculum-end 4 --curriculum-step 0.5 --curriculum-bar 0.9 --wandb --project Bimanual --name hand1_rex_attn4 --render --resume --model-path '/rl/hindsight-experience-replay/saved_models/hand2_attn4.pt'
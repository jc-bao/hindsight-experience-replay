mpirun --allow-run-as-root -np 64 python train.py \
--n-epochs 50 \
--env-name ENVNAME \
--max-trail-time 10 --trail-mode any \
--not-relabel-unmoved --random-unmoved \
--curriculum --curriculum-init 0 --curriculum-end 1 --curriculum-step 0.1 \
--curriculum-bar 0.8 \
--wandb --project PROJ --group GROUP \
--name NAME --render
mpirun --allow-run-as-root -np 64 python train.py --n-epochs 200 \
	--env-name PandaTowerBimanualOtherSide-v2 --max-trail-time 10 \
	--trail-mode any \
	--not-relabel-unmoved --random-unmoved \
	--curriculum --curriculum-bar 0.9 --curriculum-init 0.7 --curriculum-step 0.1 \
	--wandb --project Bimanual --group 2obj \
	--name os1_resume_os \
	--render --resume --model-path '/rl/hindsight-experience-replay/saved_models/PandaTowerBimanualOtherSide-v2/other_side_curri_herv2_envv21_fix_g/model.pt'
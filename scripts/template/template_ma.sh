mpirun --allow-run-as-root -np 16 \
python ../../train.py \
--env-name 'formation_hd_env' \
--actor-separated \

--num-agents 4 --dim 2 \
--learn-from-expert --collect-from-expert \
--fix-critic \
--q-coef '1.0' --imitate-coef 0 \
--actor-shared \

--resume --model-path 'LBWNB'  \

--wandb --project Formation --name hd4_separated

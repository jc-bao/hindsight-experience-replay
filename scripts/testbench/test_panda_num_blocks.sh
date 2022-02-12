python ../../train.py \
--n-epochs 10 --n-cycles 2 --n-batches 1 --buffer-size 1000 --batch-size 32 \
--env-name 'PandaTowerBimanualMaxHandover1-v1' \
--curriculum --curriculum-init 1 --curriculum-end 6 --curriculum-step 0.5 --curriculum-bar '-0.1' \
--actor-master-slave --use-attn --num-blocks 4 --shared-policy --gui
# mpirun -np 2 
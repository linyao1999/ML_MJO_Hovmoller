#!/bin/bash

# salloc --nodes 3 --qos interactive --time 04:00:00 --constraint gpu --gpus 5 --account=m4287_g

module load conda 
conda activate mlpy

for i in {1..1}; do
    for j in {1..10}; do
        exp_num=$(( (i - 1) * 10 + j ))  # Calculate experiment number
        srun --exclusive -N1 -n1 --gpus=1 env exp_num=$exp_num window_len=5 residual=true python3 best.py &
        sleep 10 
        srun --exclusive -N1 -n1 --gpus=1 env exp_num=$exp_num window_len=21 residual=true python3 best.py &
        sleep 10 
        srun --exclusive -N1 -n1 --gpus=1 env exp_num=$exp_num window_len=21 residual=false python3 best.py &
        sleep 10 
        srun --exclusive -N1 -n1 --gpus=1 env exp_num=$exp_num window_len=31 residual=true python3 best.py &
        sleep 10 
        srun --exclusive -N1 -n1 --gpus=1 env exp_num=$exp_num window_len=31 residual=false python3 best.py &
        sleep 10 
        sleep 10 
        wait
    done
    wait
done
wait
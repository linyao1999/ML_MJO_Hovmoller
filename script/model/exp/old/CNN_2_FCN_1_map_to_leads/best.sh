#!/bin/bash

# This file is to run best.py
# salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account=dasrepo_g
# salloc --nodes=4 --qos=interactive --time=04:00:00 --constraint=gpu --gpus-per-node=4 --account=dasrepo_g

# source activate python
source $(conda info --base)/etc/profile.d/conda.sh
conda activate python

for i in {1..1}; do
    for j in {1..4}; do
        exp_num=$(( (i - 1) * 4 + j ))  # Calculate the experiment number
        export exp_num 
        srun --exclusive -N1 -n1 --gpus=1 python best.py &
    done
    sleep 2  # wait 2 seconds before launching the next job
done

wait
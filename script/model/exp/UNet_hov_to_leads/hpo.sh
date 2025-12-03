#!/bin/bash

# This file is to run hpo.py
# salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account=dasrepo_g
#   salloc --nodes=2 --ntasks=2 --ntasks-per-node=1 --gpus-per-task=1 --constraint=gpu --account=m3312_g --qos=interactive --time=04:00:00

# source activate python
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mlpy

# for i in {1..4}; do
#     srun --exclusive -N1 -n1 --gpus=1 python hpo.py &
#     sleep 10  # wait 2 seconds before launching the next job
# done


# Bind one GPU per task to avoid conflicts
GPU_BIND="--gpu-bind=single:1 --cpu-bind=cores"

# Launch 16 jobs: 4 categories × 4 runs each
# for i in {1..2}; do
#   srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python3 hpo.py &
#   sleep 10
# done
srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python3 hpo.py &
sleep 100
srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python3 hpo.py &

wait






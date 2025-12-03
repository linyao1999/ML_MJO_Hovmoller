#!/bin/bash
# This script runs 16 GPU tasks (4 per node) across 4 nodes on Perlmutter
# Usage:
#   salloc --nodes=4 --ntasks=16 --ntasks-per-node=4 --gpus-per-task=1 --constraint=gpu --account=m3312_g --qos=interactive --time=04:00:00
# Then run:
#   bash run_hpo.sh

# Activate your environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mlpy

# Bind one GPU per task to avoid conflicts
GPU_BIND="--gpu-bind=single:1 --cpu-bind=cores"

# srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python hpo_mjo.py &
srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python hpo.py &
sleep 2 
srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python hpo_kelvin.py &
sleep 2
srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python hpo_rossby.py &
sleep 2
srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python hpo_gravity.py &
sleep 200

# srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python hpo_mjo.py &
srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python hpo.py &
sleep 2 
srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python hpo_kelvin.py &
sleep 2
srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python hpo_rossby.py &
sleep 2
srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python hpo_gravity.py &

# # Launch 16 jobs: 4 categories × 4 runs each
# for i in {1..2}; do
#   srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python hpo_mjo.py &
#   sleep 10
# done

# for i in {1..2}; do
#   srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python hpo_kelvin.py &
#   sleep 10
# done

# for i in {1..2}; do
#   srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python hpo_rossby.py &
#   sleep 10
# done

# for i in {1..2}; do
#   srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python hpo_gravity.py &
#   sleep 10
# done

wait

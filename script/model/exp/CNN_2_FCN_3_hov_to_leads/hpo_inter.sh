#!/bin/bash
# This script runs 16 GPU tasks (4 per node) across 4 nodes on Perlmutter
# Usage:
#   salloc --nodes=4 --ntasks=16 --ntasks-per-node=4 --gpus-per-task=1 --constraint=gpu --account=m3312_g --qos=interactive --time=04:00:00
# Then run:
#   bash run_hpo.sh

# source activate python
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mlpy

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Bind one GPU per task to avoid conflicts
GPU_BIND="--gpu-bind=single:1 --cpu-bind=cores"

srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python3 hpo.py &
sleep 2 
srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python3 hpo_mjo.py &
sleep 2 
srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python3 hpo_kelvin.py &
sleep 2
srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python3 hpo_rossby.py &
sleep 2
srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python3 hpo_gravity.py &
sleep 200

srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python3 hpo_mjo.py &
sleep 2 
srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python3 hpo_kelvin.py &
sleep 2
srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python3 hpo_rossby.py &
sleep 2
srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python3 hpo_gravity.py &
sleep 2
srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python3 hpo.py &

wait






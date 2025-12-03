#!/bin/bash

# This file is to run hpo.py
# salloc --nodes 3 --qos interactive --time 04:00:00 --constraint gpu --gpus 6 --account=m4287_g

# source activate python
# source $(conda info --base)/etc/profile.d/conda.sh
module load conda
conda activate mlpy

srun --exclusive -N1 -n1 --gpus=1 env window_len=5 residual='true' python3 hpo.py &
sleep 10  # wait 2 seconds before launching the next job

# srun --exclusive -N1 -n1 --gpus=1 env window_len=15 residual='true' python3 hpo.py &
# sleep 10  # wait 2 seconds before launching the next job

srun --exclusive -N1 -n1 --gpus=1 env window_len=5 residual='false' python3 hpo.py &
sleep 10  # wait 2 seconds before launching the next job

# srun --exclusive -N1 -n1 --gpus=1 env window_len=15 residual='false' python3 hpo.py &
# sleep 10  # wait 2 seconds before launching the next job

# srun --exclusive -N1 -n1 --gpus=1 env window_len=25 residual='true' python3 hpo.py &
# sleep 10  # wait 2 seconds before launching the next job

# srun --exclusive -N1 -n1 --gpus=1 env window_len=25 residual='false' python3 hpo.py &
# sleep 10  # wait 2 seconds before launching the next job


srun --exclusive -N1 -n1 --gpus=1 env window_len=11 residual='true' python3 hpo.py &
sleep 10  # wait 2 seconds before launching the next job

srun --exclusive -N1 -n1 --gpus=1 env window_len=21 residual='true' python3 hpo.py &
sleep 10  # wait 2 seconds before launching the next job

srun --exclusive -N1 -n1 --gpus=1 env window_len=11 residual='false' python3 hpo.py &
sleep 10  # wait 2 seconds before launching the next job

srun --exclusive -N1 -n1 --gpus=1 env window_len=21 residual='false' python3 hpo.py &
sleep 10  # wait 2 seconds before launching the next job

# srun --exclusive -N1 -n1 --gpus=1 env window_len=31 residual='true' python3 hpo.py &
# sleep 10  # wait 2 seconds before launching the next job

# srun --exclusive -N1 -n1 --gpus=1 env window_len=31 residual='false' python3 hpo.py &
sleep 10  # wait 2 seconds before launching the next job

wait






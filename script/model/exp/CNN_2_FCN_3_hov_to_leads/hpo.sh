#!/bin/bash
#SBATCH -A m3312_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=40
#SBATCH --gpus-per-node=4
#SBATCH -t 06:00:00
#SBATCH --output=outlog/hpo.out


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
# srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python3 hpo_kelvin.py &
sleep 2
srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python3 hpo_rossby.py &
sleep 2
# srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python3 hpo_gravity.py &
sleep 200

srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python3 hpo_mjo.py &
sleep 2 
# srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python3 hpo_kelvin.py &
sleep 2
srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python3 hpo_rossby.py &
sleep 2
# srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python3 hpo_gravity.py &
sleep 2
srun --exclusive -N1 -n1 --gpus-per-task=1 $GPU_BIND python3 hpo.py &

wait






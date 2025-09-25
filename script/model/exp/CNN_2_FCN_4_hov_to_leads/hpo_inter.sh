#!/bin/bash
#SBATCH -A dasrepo_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH -t 04:00:00
#SBATCH --output=outlog/hpo_%A_%a.out

# This file is to run hpo.py
# salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account=dasrepo_g

# source activate python
# source $(conda info --base)/etc/profile.d/conda.sh
module load conda
conda activate python

for i in {1..4}; do
    srun --exclusive -N1 -n1 --gpus=1 python hpo.py &
    sleep 10  # wait 2 seconds before launching the next job
done

wait






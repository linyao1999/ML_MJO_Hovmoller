#!/bin/bash
#SBATCH -A dasrepo_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH -t 06:00:00
#SBATCH --output=outlog/%j.out

# This file is to run best.py
# salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account=dasrepo_g
# salloc --nodes=4 --qos=interactive --time=04:00:00 --constraint=gpu --gpus-per-node=4 --account=dasrepo_g
if [ ! -d outlog ]; then
    mkdir outlog
fi

# source activate python
source $(conda info --base)/etc/profile.d/conda.sh
conda activate python

for i in {2..24}; do
    for j in {1..4}; do
        exp_num=$(( (i - 1) * 4 + j ))  # Calculate the experiment number
        export exp_num 
        srun --exclusive -N1 -n1 --gpus=1 python best.py &
    done
    wait
done

wait
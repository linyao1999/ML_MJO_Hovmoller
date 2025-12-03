#!/bin/bash
#SBATCH -A dasrepo
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH -t 02:00:00
#SBATCH --output=outlog/%j.out

set -x  # Enable bash debug output

# Make output directory if needed
[ -d outlog ] || mkdir outlog

# Load conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mlpy

# Run jobs in parallel, each task gets 64 CPUs
# srun --exclusive -N1 -n1 --cpus-per-task=64 python feature_maps.py &
srun --exclusive -N1 -n1 --cpus-per-task=128 python3 filter_olr_runningavg.py &

wait

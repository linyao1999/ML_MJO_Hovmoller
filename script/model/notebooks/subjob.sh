#!/bin/bash
#SBATCH -A dasrepo
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=64
#SBATCH -t 10:00:00
#SBATCH --output=outlog/%j.out

set -x  # Enable bash debug output

# Make output directory if needed
[ -d outlog ] || mkdir outlog

# Load conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate python

# Run jobs in parallel, each task gets 64 CPUs
srun --exclusive -N1 -n1 --cpus-per-task=64 python feature_maps.py &
srun --exclusive -N1 -n1 --cpus-per-task=64 python kernels.py &

wait

#!/bin/bash
#SBATCH -A dasrepo_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH -t 20:00:00
#SBATCH --output=outlog/%j.out

# salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account=dasrepo_g

# if outlog directory does not exist, create it
if [ ! -d outlog ]; then
    mkdir outlog
fi

# source activate python
source $(conda info --base)/etc/profile.d/conda.sh
conda activate python

for i in {1..4}; do
    srun --exclusive -N1 -n1 --gpus=1 python hpo.py &
    sleep 10  # wait 2 seconds before launching the next job
done

wait

for i in {1..24}; do
    for j in {1..4}; do
        exp_num=$(( (i - 1) * 4 + j ))  # Calculate the experiment number
        export exp_num 
        srun --exclusive -N1 -n1 --gpus=1 python best.py &
    done
    wait
done

wait
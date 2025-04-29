#!/bin/bash

# Example: salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account=dasrepo_g
# Example: salloc --nodes=4 --qos=interactive --time=04:00:00 --constraint=gpu --gpus-per-node=4 --account=dasrepo_g

source activate python

for exp_num in {1..16}; do
    export exp_num 
    srun --exclusive -N1 -n1 --gpus=1 python best_unet_raw30d.py &
    sleep 2  # wait 2 seconds before launching the next job
done

wait
# for exp_num in {1..16}; do
#     export exp_num 
#     srun --exclusive -N1 -n1 --gpus=1 python best_unet_raw.py &
#     sleep 2  # wait 2 seconds before launching the next job
# done

# wait

# for exp_num in {1..16}; do
#     export exp_num 
#     srun --exclusive -N1 -n1 --gpus=1 python best_unet_mjo.py &
#     sleep 2  # wait 2 seconds before launching the next job
# done

# wait

# for exp_num in {1..16}; do
#     export exp_num 
#     srun --exclusive -N1 -n1 --gpus=1 python best_unet_rossby.py &
#     sleep 2  # wait 2 seconds before launching the next job
# done

# wait

# for exp_num in {1..16}; do
#     export exp_num 
#     srun --exclusive -N1 -n1 --gpus=1 python best_unet_kelvin.py &
#     sleep 2  # wait 2 seconds before launching the next job
# done

# wait

# for exp_num in {1..16}; do
#     export exp_num 
#     srun --exclusive -N1 -n1 --gpus=1 python best_unet_gravity.py &
#     sleep 2  # wait 2 seconds before launching the next job
# done

# wait

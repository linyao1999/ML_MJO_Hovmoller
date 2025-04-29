#!/bin/bash

# Example: salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account=dasrepo_g
# Example: salloc --nodes=4 --qos=interactive --time=04:00:00 --constraint=gpu --gpus-per-node=4 --account=dasrepo_g

source activate python


for i in {1..3}; do
    srun --exclusive -N1 -n1 --gpus=1 python hpo_unet_raw.py &
    srun --exclusive -N1 -n1 --gpus=1 python hpo_unet_mjo.py &
    srun --exclusive -N1 -n1 --gpus=1 python hpo_unet_rossby.py &
    srun --exclusive -N1 -n1 --gpus=1 python hpo_unet_kelvin.py &
    srun --exclusive -N1 -n1 --gpus=1 python hpo_unet_gravity.py &
    sleep 2  # wait 2 seconds before launching the next job
done

srun --exclusive -N1 -n1 --gpus=1 python hpo_unet_rossby.py &

wait



# for i in {1..16}; do
#     srun --exclusive -N1 -n1 --gpus=1 python hpo_unet_raw.py &
#     sleep 2  # wait 2 seconds before launching the next job
# done

# wait

# for i in {1..16}; do
#     srun --exclusive -N1 -n1 --gpus=1 python hpo_unet_mjo.py &
#     sleep 2  # wait 2 seconds before launching the next job
# done

# wait

# for i in {1..16}; do
#     srun --exclusive -N1 -n1 --gpus=1 python hpo_unet_rossby.py &
#     sleep 2  # wait 2 seconds before launching the next job
# done

# wait

# for i in {1..16}; do
#     srun --exclusive -N1 -n1 --gpus=1 python hpo_unet_kelvin.py &
#     sleep 2  # wait 2 seconds before launching the next job
# done

# wait

# for i in {1..16}; do
#     srun --exclusive -N1 -n1 --gpus=1 python hpo_unet_gravity.py &
#     sleep 2  # wait 2 seconds before launching the next job
# done

# wait
#!/bin/bash

# Define the lead times
lead_times=(0 5 10 15 20 25 30)

# Define hyperparameters for HPO
learning_rates=(0.001 0.005)
batch_sizes=(32 64)
dropouts=(0.1 0.3 0.5)
epochs=(20)
optimizers=("SGD")
momentum=(0.9)
weight_decay=(0.001 0.005)
lat_ranges=(15 10)
memory_lasts=(95 29)
kernel_sizes=(25 13 7 3)
exp_nums=(1)

# Function to create a unique job script for each hyperparameter combination
submit_job() {
  lead=$1
  lr=$2
  batch_size=$3
  dropout=$4
  epochs=$5
  optimizer=$6
  momentum=$7
  weight_decay=$8
  lat_range=$9
  memory_last=${10}
  kernel_size=${11}
  exp_num=${12}

  job_name="hovmoller_l${lead}_lr${lr}_bs${batch_size}_drop${dropout}_ep${epochs}_opt${optimizer}_mom${momentum}_wd${weight_decay}_lat${lat_range}_mem${memory_last}_k${kernel_size}_exp${exp_num}"

  # Create a unique SLURM script for the job
mkdir -p ./slurm_files

# Define the script path
script="./slurm_files/run_${job_name}.slurm"

# Write the SLURM job script
cat <<EOT > ${script}
#!/bin/bash
#SBATCH -A dasrepo_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH -t 00:10:00
#SBATCH --output=outlog/$job_name.out

python /pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/single_hovmoller.py \
  --lead $lead \
  --lr $lr \
  --batch_size $batch_size \
  --dropout $dropout \
  --epochs $epochs \
  --optimizer $optimizer \
  --momentum $momentum \
  --weight_decay $weight_decay \
  --lat_range $lat_range \
  --memory_last $memory_last \
  --kernel_size $kernel_size \
  --exp_num $exp_num
EOT

  # Submit the job
  sbatch $script
  echo "Submitted job: $job_name"
}

# chmod +x ./slurm_files/*.slurm

# Iterate over all combinations of hyperparameters
for lead in "${lead_times[@]}"; do
  for lr in "${learning_rates[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
      for dropout in "${dropouts[@]}"; do
        for epoch in "${epochs[@]}"; do
          for optimizer in "${optimizers[@]}"; do
            for mom in "${momentum[@]}"; do
              for wd in "${weight_decay[@]}"; do
                for lat in "${lat_ranges[@]}"; do
                  for mem in "${memory_lasts[@]}"; do
                    for kernel in "${kernel_sizes[@]}"; do
                      for exp in "${exp_nums[@]}"; do
                        submit_job $lead $lr $batch_size $dropout $epoch $optimizer $mom $wd $lat $mem $kernel $exp
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

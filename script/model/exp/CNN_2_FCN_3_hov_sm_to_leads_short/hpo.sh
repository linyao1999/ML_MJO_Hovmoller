#!/bin/bash

# --- Configuration ---
START_LEN=5
END_LEN=31
STEP=2

# Create the output directory if it doesn't exist
mkdir -p outlog

echo "Starting job submission loop for window_len from $START_LEN to $END_LEN (step $STEP)."
echo "---"

# Loop from 3 to 31, stepping by 2
for (( window_len=$START_LEN; window_len<=$END_LEN; window_len+=$STEP )); do
    
    # Define a unique job name and file name based on the window length
    JOB_NAME="hpo_w${window_len}"
    SLURM_FILE="${JOB_NAME}.slurm"
    
    # Generate the content of the temporary SBATCH script
    cat << EOF > "${SLURM_FILE}"
#!/bin/bash
#
# Original SBATCH parameters:
#SBATCH -A m3312_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=2
#SBATCH -t 06:00:00
#SBATCH -J ${JOB_NAME}
#SBATCH --output=outlog/${JOB_NAME}_%j.out

source $(conda info --base)/etc/profile.d/conda.sh
conda activate mlpy

srun --exclusive -N1 -n1 --gpus=1 env window_len=${window_len} residual='false' python3 hpo.py &
sleep 2  # wait 2 seconds before launching the next srun (if one was defined)
srun --exclusive -N1 -n1 --gpus=1 env window_len=${window_len} residual='true' python3 hpo.py &
sleep 2  # wait 2 seconds before launching the next srun (if one was defined)

wait # Wait for the background srun process to finish

EOF

    # Submit the generated SLURM script
    echo "Submitting job: ${JOB_NAME} (window_len=${window_len})"
    sbatch "${SLURM_FILE}"
    
    # Optional: Clean up the generated file immediately after submission
    # Uncomment the line below if you don't want to keep the .slurm files
    rm "${SLURM_FILE}" 

done

echo "---"
echo "All jobs submitted. Check 'squeue -u \$USER' to monitor progress."
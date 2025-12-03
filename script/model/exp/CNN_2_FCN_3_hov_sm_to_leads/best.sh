#!/bin/bash

# --- Configuration ---
START_LEN=3
END_LEN=23
STEP=2

# Create the output directory if it doesn't exist
mkdir -p outlog

echo "Starting job submission loop for window_len from $START_LEN to $END_LEN (step $STEP)."
echo "---"

# Loop from 3 to 31, stepping by 2
for (( window_len=$START_LEN; window_len<=$END_LEN; window_len+=$STEP )); do
    
    # Define a unique job name and file name based on the window length
    JOB_NAME="bst_w${window_len}"
    SLURM_FILE="${JOB_NAME}.slurm"
    
    # Generate the content of the temporary SBATCH script
    cat << EOF > "${SLURM_FILE}"
#!/bin/bash
#SBATCH -A m3312_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=2
#SBATCH -t 01:00:00
#SBATCH -J ${JOB_NAME}
#SBATCH --output=outlog/${JOB_NAME}_%j.out

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate mlpy

for i in {1..1}; do
    for j in {1..8}; do
        exp_num=\$(( (i - 1) * 16 + j ))  # Calculate experiment number
        srun --exclusive -N1 -n1 --gpus=1 \\
            env exp_num=\$exp_num window_len=${window_len} residual=false \\
            python3 best.py &
    done
    wait
done

wait # Wait for the background srun processes to finish
EOF

    # Submit the generated SLURM script
    echo "Submitting job: ${JOB_NAME} (window_len=${window_len})"
    sbatch "${SLURM_FILE}"
    
    # Optional: Clean up the generated file immediately after submission
    rm "${SLURM_FILE}"

done

echo "---"
echo "All jobs submitted. Check 'squeue -u \$USER' to monitor progress."

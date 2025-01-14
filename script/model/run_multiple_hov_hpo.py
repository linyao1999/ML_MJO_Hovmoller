from multiprocessing import Semaphore, Pool
import subprocess

# Allocate interactive session
# Example: salloc --nodes 1 --qos interactive --time 02:00:00 --constraint gpu --gpus 4 --account=m4736_g

max_jobs = 4  # Limit concurrent jobs to 4 (adjust based on resources)
semaphore = Semaphore(max_jobs)

# Define the lead times
lead_times = list(range(0, 31, 5))  # [0, 5, 10, ..., 30]

# Define hyperparameters for HPO
learning_rates = [0.001, 0.005]
batch_sizes = [32, 64]
dropouts = [0.1, 0.2]
epochs = [10, 20]
optimizers = ["SGD", "Adam"]
momentum = [0.9]
weight_decay = [0.0001, 0.001]
lat_ranges = [10, 20]
memory_lasts = [5, 10]
kernel_sizes = [3, 5]

# Generate all combinations of hyperparameters
from itertools import product

hyperparameter_combinations = list(product(
    lead_times, learning_rates, batch_sizes, dropouts, epochs, optimizers,
    momentum, weight_decay, lat_ranges, memory_lasts, kernel_sizes
))

# Function to run a single job
def run_job(params):
    lead, lr, batch_size, dropout, epochs, optimizer, momentum, wd, lat_range, memory_last, kernel_size = params
    command = (
        f"python single_hovmoller.py "
        f"--lead {lead} "
        f"--lr {lr} "
        f"--batch_size {batch_size} "
        f"--dropout {dropout} "
        f"--epochs {epochs} "
        f"--optimizer {optimizer} "
        f"--momentum {momentum} "
        f"--weight_decay {wd} "
        f"--lat_range {lat_range} "
        f"--memory_last {memory_last} "
        f"--kernel_size {kernel_size}"
    )
    with semaphore:
        print(f"Starting job with command: {command}")
        process = subprocess.Popen(command, shell=True)
        process.wait()

# Run jobs in parallel
with Pool(max_jobs) as pool:
    pool.map(run_job, hyperparameter_combinations)

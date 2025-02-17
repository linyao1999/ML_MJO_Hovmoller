from multiprocessing import Semaphore
import subprocess

# salloc --nodes 1 --qos interactive --time 03:10:00 --constraint gpu --gpus 4 --account=m3312_g

max_jobs = 4  # Limit concurrent jobs to 4 (adjust based on resources)
semaphore = Semaphore(max_jobs)
lead_times = list(range(0, 31, 5))  # [0, 5, 10, ..., 30]

def run_job(lead):
    command = f"python single_hovmoller.py --lead {lead}"
    with semaphore:
        print(f"Starting job for lead={lead} with command: {command}")
        process = subprocess.Popen(command, shell=True)
        process.wait()

from multiprocessing import Pool
with Pool(max_jobs) as pool:
    pool.map(run_job, lead_times)

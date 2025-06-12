# srun -n 128 python bccs.py
# bccs.py

from mpi4py import MPI
import numpy as np
import os
import yaml
import sys
from pathlib import Path
src_utils_path = Path("../src/utils")
sys.path.append(str(src_utils_path))

def chunkify(lst, n):
    """Splits lst into n roughly equal chunks."""
    return [lst[i::n] for i in range(n)]

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    exp_total = 96
    exp_dir = "/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/exp"
    exp_names = [
        "CNN_1_FCN_1_map_to_leads",
        "CNN_2_FCN_2_map_to_leads",
        "CNN_3_FCN_3_map_to_leads",
        "CNN_2_FCN_2_hov_to_leads",
        "CNN_2_FCN_2_hov_two_to_leads",
        "CNN_2_FCN_2_latavg_to_leads",
    ]

    for folder in [os.path.join(exp_dir, name) for name in exp_names]:
        config_path = os.path.join(folder, 'best_config.yaml')
        if not os.path.exists(config_path):
            if rank == 0:
                print(f"{config_path} does not exist. Skipping...")
            continue
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        lat_range = config['data']['lat_range']
        lead = config['data']['lead']
        memory_last = config['data']['memory_last']
        output_path = config['prediction_save_path']

        fn_list = [output_path.replace("exp1/", f"exp{exp_num}/") for exp_num in range(1, exp_total+1)]
        exp_nums = list(range(1, exp_total+1))

        # Split jobs among ranks
        local_fns = chunkify(fn_list, size)[rank]
        local_exp_nums = chunkify(exp_nums, size)[rank]

        # Each rank processes its assigned jobs
        local_results = []
        for fn, exp_num in zip(local_fns, local_exp_nums):
            try:
                # Test period
                bcc, rmse = metrics.get_skill_one_all_leads(
                    'ROMI', fn, rule='Iamp>1.0', month_list=None,
                    datesta='2016-01-01', dateend='2021-12-31', lead_max=lead,
                    Fnmjo=config["data"]["target_path"]
                )
                # Validation period
                bccv, rmsev = metrics.get_skill_one_all_leads(
                    'ROMI', fn, rule='Iamp>1.0', month_list=None,
                    datesta='2010-01-01', dateend='2015-12-31', lead_max=lead,
                    Fnmjo=config["data"]["target_path"]
                )
                local_results.append((exp_num, bcc, bccv, rmse, rmsev))
            except Exception as e:
                print(f"[Rank {rank}] Failed {fn}: {e}")
                local_results.append((exp_num, None, None, None, None))

        # Gather all results at root
        all_results = comm.gather(local_results, root=0)

        if rank == 0:
            # Flatten and sort by exp_num
            flat_results = [item for sublist in all_results for item in sublist]
            flat_results.sort(key=lambda x: x[0])

            # Convert to arrays, keep names as in your original code
            bccs   = np.array([x[1] if x[1] is not None else np.nan for x in flat_results])
            bccvs  = np.array([x[2] if x[2] is not None else np.nan for x in flat_results])
            rmses  = np.array([x[3] if x[3] is not None else np.nan for x in flat_results])
            rmsevs = np.array([x[4] if x[4] is not None else np.nan for x in flat_results])

            # calcualte ensemble-mean BCC and RMSE
            bcc, rmse = metrics.get_skill_all_leads_ensemble_mean(
                fn_list=fn_list,
                exp_num_list=np.arange(1, exp_total+1),
                lat_lim=lat_range,
                leadmjo=lead,
                datesta='2016-01-01',
                dateend='2021-12-31',
                ampthred=1,
                Fnmjo=config["data"]["target_path"]
            )

            bccv, rmsev = metrics.get_skill_all_leads_ensemble_mean(
                fn_list=fn_list,
                exp_num_list=np.arange(1, exp_total+1),
                lat_lim=lat_range,
                leadmjo=lead,
                datesta='2010-01-01',
                dateend='2015-12-31',
                ampthred=1,
                Fnmjo=config["data"]["target_path"]
            )

            # Save
            output_dir = Path(folder)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / 'skills_mpi.npz'
            np.savez_compressed(
                output_file, 
                bccs=bccs, 
                bccvs=bccvs, 
                rmses=rmses, 
                rmsevs=rmsevs,
                bcc=bcc,
                rmse=rmse,
                bccv=bccv,
                rmsev=rmsev
            )
            print(f"Saved: {output_file}")

if __name__ == "__main__":
    main()

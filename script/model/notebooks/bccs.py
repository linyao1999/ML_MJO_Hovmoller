import sys
from pathlib import Path
src_utils_path = Path("../src/utils")
sys.path.append(str(src_utils_path))
import metrics as mjo
import yaml  
import numpy as np
import xarray as xr 
import os
import re

# salloc --nodes 1 --qos interactive --time 04:00:00 --constraint cpu --account=dasrepo

exp_total = 96

exp_dir = "/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/exp"
# folder_list = [
#     os.path.join(exp_dir, name)
#     for name in os.listdir(exp_dir)
#     if os.path.isdir(os.path.join(exp_dir, name))
# ]

exp_names = [
    # "CNN_1_FCN_1_map_to_leads",
    # "CNN_2_FCN_2_map_to_leads",
    # "CNN_3_FCN_3_map_to_leads",
    # "CNN_2_FCN_2_hov_to_leads",
    # "CNN_2_FCN_2_hov_two_to_leads",
    # "CNN_2_FCN_2_latavg_to_leads",
    # "CNN_2_FCN_2_latavg_time_to_leads",
    "CNN_2_FCN_2_map_sym_to_leads",
    "CNN_2_FCN_2_map_asym_to_leads",
]

folder_list = [os.path.join(exp_dir, name) for name in exp_names]

print(folder_list)

for folder in folder_list:
    output_dir = folder
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist. Skipping...")
        continue

    config_path = os.path.join(output_dir, 'best_config.yaml')
    if not os.path.exists(config_path):
        print(f"{config_path} does not exist. Skipping...")
        continue

    print(f"Processing folder: {folder}")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    lat_range = config['data']['lat_range']
    lead = config['data']['lead']
    memory_last = config['data']['memory_last']
    output_path = config['prediction_save_path']

    fn_list = []
    for exp_num in range(1, exp_total+1):
        fn = re.sub(r"exp\d+/", f"exp{exp_num}/", output_path)
        fn_list.append(fn)

    # print the current time 
    print(f"Current time: {np.datetime64('now')}")
    bcc, rmse = mjo.get_skill_all_leads_parallel(
        'ROMI',
        fn_list=fn_list,
        rule='Iamp>1.0',
        month_list=None,
        datesta='2016-01-01',
        dateend='2021-12-31',
        lead_max=lead,
        exp_list=np.arange(1, exp_total+1),
        Fnmjo=config["data"]["target_path"]
    )
    # print the current time
    print(f"Current time: {np.datetime64('now')}")
    bccv, rmsev = mjo.get_skill_all_leads_parallel(
        'ROMI',
        fn_list=fn_list,
        rule='Iamp>1.0',
        month_list=None,
        datesta='2010-01-01',
        dateend='2015-12-31',
        lead_max=lead,
        exp_list=np.arange(1, exp_total+1),
        Fnmjo=config["data"]["target_path"]
    )
    # print the current time
    print(f"Current time: {np.datetime64('now')}")

    bccs = []
    bccvs = []
    rmses = []
    rmsevs = []
    for exp_num in range(1, exp_total+1):
        bccs.append(bcc.get(exp_num, np.nan))
        bccvs.append(bccv.get(exp_num, np.nan))
        rmses.append(rmse.get(exp_num, np.nan))
        rmsevs.append(rmsev.get(exp_num, np.nan))

    bccs = np.array(bccs)
    bccvs = np.array(bccvs)
    rmses = np.array(rmses)
    rmsevs = np.array(rmsevs)

    # calcualte ensemble-mean BCC and RMSE
    bcc, rmse = mjo.get_skill_all_leads_ensemble_mean(
        fn_list=fn_list,
        exp_num_list=np.arange(1, exp_total+1),
        lat_lim=lat_range,
        leadmjo=lead,
        datesta='2016-01-01',
        dateend='2021-12-31',
        ampthred=1,
        Fnmjo=config["data"]["target_path"]
    )

    bccv, rmsev = mjo.get_skill_all_leads_ensemble_mean(
        fn_list=fn_list,
        exp_num_list=np.arange(1, exp_total+1),
        lat_lim=lat_range,
        leadmjo=lead,
        datesta='2010-01-01',
        dateend='2015-12-31',
        ampthred=1,
        Fnmjo=config["data"]["target_path"]
    )

    # Save the results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'skills.npz'
    np.savez_compressed(output_file, bccs=bccs, bccvs=bccvs, rmses=rmses, rmsevs=rmsevs, bcc=bcc, rmse=rmse, bccv=bccv, rmsev=rmsev)

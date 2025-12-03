# this script is to calculate the power spectrum of the last convolutional layer 
# and save it to a file
import sys
from pathlib import Path
src_utils_path = Path("../src")
sys.path.append(str(src_utils_path))
import utils.feature_maps as mjo
import yaml  
import numpy as np
import xarray as xr 
import os
import re
import gc
# salloc --nodes 1 --qos interactive --time 04:00:00 --constraint cpu --account=dasrepo


window_len = 21
residual = 'False'

exp_total = 10
hidden_layer = [-1,]
hid_str = '_'.join([str(x) for x in hidden_layer])
c = 51
relu = False  # whether to get the power norm after ReLU
if relu:
    reluflg = '_relu'
else:
    reluflg = ''
exp_dir = "/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/exp"
exp_names = [
    "CNN_2_FCN_3_hov_sm_to_leads",
]
folder_list = [os.path.join(exp_dir, name) for name in exp_names]
# print(folder_list)

for folder in folder_list:
    power_norm = []
    output_dir = folder
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist. Skipping...")
        continue

    config_path = os.path.join(output_dir, f'./yaml/best_config_sm{window_len}_res{residual}.yaml')
    if not os.path.exists(config_path):
        print(f"{config_path} does not exist. Skipping...")
        continue

    print(f"Processing folder: {folder}")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    mjo.get_hid_fftpower_norm_ensemble(
        config,
        exp_num_list=np.arange(1, exp_total+1),
        hidden_layer=hidden_layer,
        after_relu=relu,
    )

    # read the calculated power norm files
    feature_path = config['prediction_save_path'].replace('.nc', f'_power_norm_ch{hid_str}{reluflg}.nc')
    for exp_num in range(1, exp_total+1):
        fn = re.sub(r"exp\d+/", f"exp{exp_num}/", feature_path)
        da = xr.open_dataarray(fn)
        power_norm.append(da)  # [c, m, k]
    power_norm = xr.concat(power_norm, dim='exp_num')  # [exp_num, c, m, k]
    # power_norm = power_norm.mean(dim='exp_num')  # average over the ensemble
    power_norm.name = 'power_norm'
    power_norm.attrs['hidden_layer'] = hid_str

    # save the averaged power norm to a new file
    out_fn = output_dir + f'/feature_power_norm_{hid_str}{reluflg}_sm{str(window_len)}_res{str(residual)}.nc'
    power_norm.to_netcdf(out_fn, mode='w')
    print(f'Saved to {out_fn}')
    print('Done!')

    del power_norm, da
    gc.collect()





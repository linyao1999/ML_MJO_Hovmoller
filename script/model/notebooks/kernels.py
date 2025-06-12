# this script is to calculate the power spectrum of the last convolutional layer 
# and save it to a file
import sys
from pathlib import Path
src_utils_path = Path("../src/utils")
sys.path.append(str(src_utils_path))
import feature_maps as mjo
import yaml  
import numpy as np
import xarray as xr 
import os
import re

exp_total = 96
hidden_layer = ['cnn.network.3',]
hid_str = '_'.join(hidden_layer)
c = 51

exp_dir = "/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/exp"
exp_names = [
    "CNN_1_FCN_1_map_to_leads",
    "CNN_2_FCN_2_map_to_leads",
    "CNN_3_FCN_3_map_to_leads",
    "CNN_2_FCN_2_hov_to_leads",
    "CNN_2_FCN_2_hov_two_to_leads",
    "CNN_2_FCN_2_latavg_to_leads",
    # "CNN_2_FCN_2_latavg_time_to_leads",
    "CNN_2_FCN_2_map_sym_to_leads",
    "CNN_2_FCN_2_map_asym_to_leads",
]
folder_list = [os.path.join(exp_dir, name) for name in exp_names]
print(folder_list)

for folder in folder_list:
    power_norm = []
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

    mjo.get_kernel_power_mk_one(
        config,
        exp_num_list=np.arange(1, exp_total+1),
        hidden_layer=hidden_layer,
    )

    # read the calculated power norm files
    feature_path = config['prediction_save_path'].replace('.nc', '_kernels.npz')
    for exp_num in range(1, exp_total+1):
        fn = re.sub(r"exp\d+/", f"exp{exp_num}/", feature_path) # '.npz'
        # read npz
        data = np.load(fn, allow_pickle=True)
        norm_powers = data["norm_powers"]  # [out, in, ky, kx] (as list of arrays, or array with dtype=object)
        kx = data["kx"]
        ky = data["ky"]

        # Stack to array if needed (if loaded as list of arrays)
        if isinstance(norm_powers, np.ndarray) and norm_powers.dtype == object:
            norm_powers = np.stack(norm_powers)  # shape: [out, in, ky, kx]

        # Reshape to [filter, ky, kx]
        out, in_ch, ky_len, kx_len = norm_powers.shape
        norm_powers_reshaped = norm_powers.reshape(out * in_ch, ky_len, kx_len)

        # Create DataArray
        da = xr.DataArray(
            norm_powers_reshaped,
            dims=("filter", "ky", "kx"),
            coords={
                "filter": np.arange(norm_powers_reshaped.shape[0]),
                "ky": ky,
                "kx": kx,
            },
            name="norm_power"
        )
        power_norm.append(da)  # [c, m, k]

    power_norm = xr.concat(power_norm, dim='exp_num') # [exp_num, c, m, k]
    # power_norm = power_norm.mean(dim='exp_num')  # average over the ensemble
    power_norm.name = 'power_norm'
    power_norm.attrs['hidden_layer'] = hid_str

    # save the averaged power norm to a new file
    out_fn = output_dir + f'/kernels_power_norm_{hid_str}_c{c}{reluflg}.nc'
    power_norm.to_netcdf(out_fn, mode='w')
    print(f'Saved to {out_fn}')
    print('Done!')

    



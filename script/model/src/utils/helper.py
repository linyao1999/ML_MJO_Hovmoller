import numpy as np 
import xarray as xr
import os
import fnmatch


def find_missing_file(base_dir, exp_list=['1',], target_fn_lists=[]):
    missing_files = []
    
    for exp_num in exp_list:
        exp_dir = os.path.join(base_dir, f"exp{exp_num}")
        # print(f"Checking experiment directory: {exp_dir}")
        if not os.path.exists(exp_dir):
            print(f"Experiment directory not found: {exp_dir}")
            break  
        
        for fn in target_fn_lists:
            file_found = None
            # print(f"Looking for files with lead {lead} in {exp_dir}")
            for file in os.listdir(exp_dir):
                # Use fnmatch for pattern matching
                if fnmatch.fnmatch(file, fn):
                    file_found = os.path.join(exp_dir, file)
                    # print(f"Matched file: {file_found}")
                    break
            
            if not file_found:
                missing_files.append(fn)
    
    return missing_files

def generate_fn_list(
    lead_list=[0,],
    lat_ranges = [10, 15],
    learning_rates=[0.001, 0.005],
    batch_sizes=[32, 64],
    dropouts=[0.1, 0.3, 0.5],
    epochs=[20,],
    optimizers=["SGD",],
    momentum=[0.9,],
    weight_decay=[0.001, 0.005],
    memory_lasts=[95, 29],
    kernel_sizes=[25, 13, 7, 3],
    channels_list_strs=["32_8",],
    hidden_layers_strs=["1024_128",]):

    fn_list = []

    for lat in lat_ranges:
        for lr in learning_rates:
            for bs in batch_sizes:
                for do in dropouts:
                    for ep in epochs:
                        for opt in optimizers:
                            for mom in momentum:
                                for wd in weight_decay:
                                    for ml in memory_lasts:
                                        for ks in kernel_sizes:
                                            for channels_list_str in channels_list_strs:
                                                for hidden_layers_str in hidden_layers_strs:
                                                    for lead in lead_list:
                                                        fn = f"OLR_{lat}deg_lead{lead}_lr{lr}_batch{bs}_dropout{do}_ch_{channels_list_str}_ksize_{ks}_hidden_{hidden_layers_str}_opt_{opt}_mom{mom}_wd{wd}_mem{ml}.nc"
                                                        fn_list.append(fn)

    return fn_list
    

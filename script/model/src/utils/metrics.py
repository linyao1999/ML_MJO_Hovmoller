import numpy as np 
import xarray as xr
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import fnmatch


def bulk_bcc(F, O):
    # F: forecast [time, index]
    # O: observation [time, index]

    # calculate the correlation coefficient
    corr_nom = sum(F[:,0]*O[:,0] + F[:,1]*O[:,1])
    corr_denom = np.sqrt(sum(F[:,0]**2 + F[:,1]**2)*sum(O[:,0]**2 + O[:,1]**2))

    return corr_nom/corr_denom

def bulk_rmse(F, O):
    # F: forecast [time, index]
    # O: observation [time, index]

    # calculate the correlation coefficient
    rmse = np.sqrt(np.mean( (F[:,0]-O[:,0])**2 + (F[:,1]-O[:,1])**2 ))

    return rmse

def amp_error(F, O):
    # F: forecast [time, index]
    # O: observation [time, index]

    AF = np.sqrt(F[:,0]**2 + F[:,1]**2)
    AO = np.sqrt(O[:,0]**2 + O[:,1]**2)

    amp_err = np.mean(AF-AO)

    return amp_err

def vectorized_get_phase(RMM1, RMM2):
    # RMM1 and RMM2 are 1D arrays
    phase = np.zeros_like(RMM1)  # Initialize the phase array with zeros

    phase = np.where((RMM1 >= 0) & (RMM2 >= 0) & (RMM1 >= RMM2), 5, phase)
    phase = np.where((RMM1 >= 0) & (RMM2 >= 0) & (RMM1 <= RMM2), 6, phase)
    phase = np.where((RMM1 <= 0) & (RMM2 >= 0) & (-RMM1 <= RMM2), 7, phase)
    phase = np.where((RMM1 <= 0) & (RMM2 >= 0) & (-RMM1 >= RMM2), 8, phase)
    phase = np.where((RMM1 <= 0) & (RMM2 <= 0) & (RMM1 <= RMM2), 1, phase)
    phase = np.where((RMM1 <= 0) & (RMM2 <= 0) & (RMM1 >= RMM2), 2, phase)
    phase = np.where((RMM1 >= 0) & (RMM2 <= 0) & (RMM1 <= -RMM2), 3, phase)
    phase = np.where((RMM1 >= 0) & (RMM2 <= 0) & (RMM1 >= -RMM2), 4, phase)

    return phase

def get_phase_amp(mjo_ind, datasta, dataend, 
                  Fnmjo = '/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/target/romi/ROMI_NOAA_1979to2022.nc'): # get initial phase and amplitude

    ds0 = xr.open_dataset(Fnmjo).sel(time=slice(datasta, dataend))
    ds = ds0

    if mjo_ind == 'RMM':
        phase = vectorized_get_phase(ds['RMM'][:,0].values, ds['RMM'][:,1].values)
    elif mjo_ind == 'ROMI':
        phase = vectorized_get_phase(ds.ROMI[:,1].values, -ds['ROMI'][:,0].values)

    amp = np.sqrt(ds[mjo_ind][:,0].values**2+ds[mjo_ind][:,1].values**2)

    return phase, amp

def get_skill_one(mjo_ind, fn, rule='Iamp>1.0', month_list=None,
                       Fnmjo = '/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/target/romi/ROMI_NOAA_1979to2022.nc'):
    # mjo_ind: the index of MJO
    # fn: prediction file 
    # rule: the rule to select the data
    # month_list: the list of months to select the data
    # Fnmjo: original target file 

    ds = xr.open_dataset(fn)
    datesta = ds.time[0].values
    dateend = ds.time[-1].values

    phase, amp = get_phase_amp(mjo_ind=mjo_ind, datasta=datesta, dataend=dateend, Fnmjo=Fnmjo)
    ds['iphase'] = xr.DataArray(phase, dims=['time'], attrs={'long_name': 'initial phase of MJO'})
    ds['iamp'] = xr.DataArray(amp, dims=['time'], attrs={'long_name': 'initial amplitude of MJO'})
    # # target phase and amplitude
    # phase = vectorized_get_phase(ds[mjo_ind+'t'][:,0].values, ds[mjo_ind+'t'][:,1].values)
    # amp = np.sqrt(ds[mjo_ind+'t'][:,0].values**2+ds[mjo_ind+'t'][:,1].values**2)
    # ds['tphase'] = xr.DataArray(phase, dims=['time'], attrs={'long_name': 'target phase of MJO'})
    # ds['tamp'] = xr.DataArray(amp, dims=['time'], attrs={'long_name': 'target amplitude of MJO'})
    
    if rule == 'Iamp>1.0':
        ds_sel = ds.where(ds.iamp>1.0, drop=True)
    elif rule == 'Tamp>1.0':
        ds_sel = ds.where(ds.tamp>1.0, drop=True)
    elif rule == 'DJFM':
        ds_sel = ds.where(ds.time.dt.month.isin([12,1,2,3]), drop=True)
    elif rule == 'DJFM+Iamp>1.0':
        ds_sel = ds.where(ds.time.dt.month.isin([12,1,2,3]), drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp>1.0, drop=True)
    elif rule == 'month+Iamp>1.0':
        ds_sel = ds.where(ds.time.dt.month.isin(month_list), drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp>1.0, drop=True)
    elif rule == '1-1.5':
        ds_sel = ds.where(ds.time.dt.month.isin([10,11,12,1,2,3]), drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp>1.0, drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp<=1.5, drop=True)
    elif rule == '1.5-2':
        ds_sel = ds.where(ds.time.dt.month.isin([10,11,12,1,2,3]), drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp>1.5, drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp<=2.0, drop=True)
    elif rule == '2-4':
        ds_sel = ds.where(ds.time.dt.month.isin([10,11,12,1,2,3]), drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp>2.0, drop=True)
    elif rule == '0-1':
        ds_sel = ds.where(ds.time.dt.month.isin([10,11,12,1,2,3]), drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp<1.0, drop=True)
    elif rule == 'None':
        ds_sel = ds

    bcc = bulk_bcc(ds_sel['predictions'], ds_sel['targets'])
    rmse = bulk_rmse(ds_sel['predictions'], ds_sel['targets'])

    return bcc, rmse

def compute_get_skill_one(mjo_ind, fn, rule='Iamp>1.0', month_list=None, lead=0, exp_num='1',
                          Fnmjo='/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/target/romi/ROMI_NOAA_1979to2022.nc'):    
    bcc, rmse = get_skill_one(mjo_ind, fn, rule=rule, month_list=month_list, Fnmjo=Fnmjo)
    return (lead, exp_num), {'bcc': bcc, 'rmse': rmse}

def get_skill_parallel(mjo_ind, fn_list={}, rule='Iamp>1.0', month_list=None, lead_list=[0,], exp_list=['1',],
                       Fnmjo='/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/target/romi/ROMI_NOAA_1979to2022.nc'):
    """
    Calculate skills for different lead times and experiment numbers in parallel.
    
    Parameters:
    - mjo_ind: The index of MJO
    - fn_list: Dictionary containing file handles indexed by (lead, exp_num)
    - rule: A rule to filter data, default is 'Iamp>1.0'
    - month_list: List of months to consider, default is None (use all months)
    - lead_list: List of lead times
    - exp_list: List of experiment numbers
    - Fnmjo: Path to the target ROMI dataset
    
    Returns:
    - bcc_list: Dictionary of bcc values indexed by (lead, exp_num)
    - rmse_list: Dictionary of rmse values indexed by (lead, exp_num)
    """
    
    bcc_list = {}
    rmse_list = {}

    with ProcessPoolExecutor() as executor:
        # Submit tasks to the executor
        futures = [
            executor.submit(compute_get_skill_one, mjo_ind, fn=fn_list[(lead, exp_num)], rule=rule, 
                            month_list=month_list, lead=lead, exp_num=exp_num, Fnmjo=Fnmjo)
            for lead in lead_list for exp_num in exp_list
        ]
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            (lead, exp_num), result = future.result()
            bcc_list[(lead, exp_num)] = result['bcc']
            rmse_list[(lead, exp_num)] = result['rmse']
                
    return bcc_list, rmse_list

def generate_fn_list(base_dir, lead_list=[0,], exp_list=['1',], lat=20, lr=0.01, batch_size=64, mem=29):
    fn_list = {}
    
    for exp_num in exp_list:
        exp_dir = os.path.join(base_dir, f"exp{exp_num}")
        # print(f"Checking experiment directory: {exp_dir}")
        if not os.path.exists(exp_dir):
            print(f"Experiment directory not found: {exp_dir}")
            break  
        
        for lead in lead_list:
            file_found = None
            # print(f"Looking for files with lead {lead} in {exp_dir}")
            for file in os.listdir(exp_dir):
                # Use fnmatch for pattern matching
                if fnmatch.fnmatch(file, f"*{lat}deg*lead{lead}*lr{lr}*batch{batch_size}*mem{mem}.nc"):
                    file_found = os.path.join(exp_dir, file)
                    # print(f"Matched file: {file_found}")
                    break
            
            if file_found:
                fn_list[(lead, exp_num)] = file_found
            else:
                print(f"No matching file for lead {lead}, experiment {exp_num} in {exp_dir}")
    
    return fn_list
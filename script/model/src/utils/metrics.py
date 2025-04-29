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

def get_skill_one(mjo_ind, fn, rule='Iamp>1.0', month_list=None, datesta=None, dateend=None,
                       Fnmjo = '/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/target/romi/ROMI_NOAA_1979to2022.nc'):
    # mjo_ind: the index of MJO
    # fn: prediction file 
    # rule: the rule to select the data
    # month_list: the list of months to select the data
    # Fnmjo: original target file 

    ds0 = xr.open_dataset(fn)
    if datesta is None:
        datesta = ds0.time[0].values
    else:
        # find the latest datesta
        datesta = max(np.datetime64(datesta), np.datetime64(ds0.time[0].values))


    if dateend is None:
        dateend = ds0.time[-1].values
    else:
        # find the earliest dateend
        dateend = min(np.datetime64(dateend), np.datetime64(ds0.time[-1].values))

    ds = ds0.sel(time=slice(datesta, dateend))
    # print('time: ', ds.dims['time'])
    # print('datesta: ', datesta)
    # print('dateend: ', dateend)
    phase, amp = get_phase_amp(mjo_ind=mjo_ind, datasta=datesta, dataend=dateend, Fnmjo=Fnmjo)
    # print('phase: ', phase.shape)
    ds['iphase'] = xr.DataArray(phase, dims=['time'], attrs={'long_name': 'initial phase of MJO'})
    ds['iamp'] = xr.DataArray(amp, dims=['time'], attrs={'long_name': 'initial amplitude of MJO'})
    # # target phase and amplitude
    # phase = vectorized_get_phase(ds['targets'][:,0].values, ds['targets'][:,1].values)
    # amp = np.sqrt(ds['targets'][:,0].values**2+ds['targets'][:,1].values**2)
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

def compute_get_skill_one(mjo_ind, fn, rule='Iamp>1.0', month_list=None, lead=0, exp_num='1', datesta='2015-01-01', dateend='2018-12-31',
                          Fnmjo='/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/target/romi/ROMI_NOAA_1979to2022.nc'):    
    bcc, rmse = get_skill_one(mjo_ind, fn, rule=rule, month_list=month_list, Fnmjo=Fnmjo, datesta=datesta, dateend=dateend)
    return (lead, exp_num), {'bcc': bcc, 'rmse': rmse}

def get_skill_parallel(mjo_ind, datesta='2015-01-01', dateend='2018-12-31', fn_list={}, rule='Iamp>1.0', month_list=None, lead_list=[0,], exp_list=['1',],
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
                            month_list=month_list, lead=lead, exp_num=exp_num, Fnmjo=Fnmjo, datesta=datesta, dateend=dateend)
            for lead in lead_list for exp_num in exp_list
        ]
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            (lead, exp_num), result = future.result()
            bcc_list[(lead, exp_num)] = result['bcc']
            rmse_list[(lead, exp_num)] = result['rmse']
                
    return bcc_list, rmse_list

def generate_fn_list(base_dir, lead_list=[0,], exp_list=['1',], lat=15, fileflg='*.nc'):
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
                if fnmatch.fnmatch(file, f"OLR_{lat}deg_lead{lead}_{fileflg}"):
                    file_found = os.path.join(exp_dir, file)
                    # print(f"Matched file: {file_found}")
                    break
            
            if file_found:
                fn_list[(lead, exp_num)] = file_found
            else:
                print(f"No matching file for lead {lead}, experiment {exp_num} in {exp_dir}")
    
    return fn_list


def generate_fn_list_hpo(
    base_dir='/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/predictions/hovmoller/epo20/exp1',
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
                                                        fn = f"{base_dir}/OLR_{lat}deg_lead{lead}_lr{lr}_batch{bs}_dropout{do}_ch_{channels_list_str}_ksize_{ks}_hidden_{hidden_layers_str}_opt_{opt}_mom{mom}_wd{wd}_mem{ml}.nc"
                                                        fn_list.append(fn)

    return fn_list
    
def compute_skill_for_hpo(args):
    mjo_ind, fn, rule, month_list, datesta, dateend = args
    bcc, rmse = get_skill_one(mjo_ind, fn, rule=rule, month_list=month_list, datesta=datesta, dateend=dateend)
    return {fn: bcc}, {fn: rmse}

def get_skill_hpo_exp1(
    mjo_ind='ROMI',
    datesta='2015-01-01',
    dateend='2018-12-31',
    rule= 'Iamp>1.0',
    month_list=None,
    base_dir='/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/predictions/hovmoller/epo20/exp',
    exp = 1,
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

    bcc_list = {}
    rmse_list = {}

    for lead in lead_list:
        # Generate the list of file paths for HPO configurations at a given lead time and experiment number
        fn_list_hpo = generate_fn_list_hpo(base_dir=f'{base_dir}{exp}', lead_list=[lead,], lat_ranges=lat_ranges, learning_rates=learning_rates,
                                            batch_sizes=batch_sizes, dropouts=dropouts, epochs=epochs, optimizers=optimizers,
                                            momentum=momentum, weight_decay=weight_decay, memory_lasts=memory_lasts, kernel_sizes=kernel_sizes,
                                            channels_list_strs=channels_list_strs, hidden_layers_strs=hidden_layers_strs)    


        bcc_list[(lead, exp)] = []
        rmse_list[(lead, exp)] = []

        # Use ProcessPoolExecutor to parallelize the computation
        with ProcessPoolExecutor() as executor:
            args_list = [(mjo_ind, fn, rule, month_list, datesta, dateend) for fn in fn_list_hpo]
            results = list(executor.map(compute_skill_for_hpo, args_list))

        # Collect the results from the parallel computation
        for bcc, rmse in results:
            bcc_list[(lead, exp)].append(bcc)
            rmse_list[(lead, exp)].append(rmse)

    return bcc_list, rmse_list


def get_skill_one_all_leads(mjo_ind, fn, rule='Iamp>1.0', month_list=None, datesta=None, dateend=None, lead_max=30,
                       Fnmjo = '/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/target/romi/ROMI_NOAA_1979to2022.nc'):
    # mjo_ind: the index of MJO
    # fn: prediction file 
    # rule: the rule to select the data
    # month_list: the list of months to select the data
    # Fnmjo: original target file 

    ds0 = xr.open_dataset(fn)
    if datesta is None:
        datesta = ds0.time[0].values
    else:
        # find the latest datesta
        datesta = max(np.datetime64(datesta), np.datetime64(ds0.time[0].values))


    if dateend is None:
        dateend = ds0.time[-1].values
    else:
        # find the earliest dateend
        dateend = min(np.datetime64(dateend), np.datetime64(ds0.time[-1].values))

    ds = ds0.sel(time=slice(datesta, dateend))

    phase, amp = get_phase_amp(mjo_ind=mjo_ind, datasta=datesta, dataend=dateend, Fnmjo=Fnmjo)

    ds['iphase'] = xr.DataArray(phase, dims=['time'], attrs={'long_name': 'initial phase of MJO'})
    ds['iamp'] = xr.DataArray(amp, dims=['time'], attrs={'long_name': 'initial amplitude of MJO'})

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

    bcc = np.zeros(lead_max+1)
    rmse = np.zeros(lead_max+1)

    for i, lead in enumerate(range(lead_max+1)):  
        bcc[i] = bulk_bcc(ds_sel['predictions'][:,2*i:2*i+2], ds_sel['targets'][:,2*i:2*i+2])
        rmse[i] = bulk_rmse(ds_sel['predictions'][:,2*i:2*i+2], ds_sel['targets'][:,2*i:2*i+2])

    return bcc, rmse

def compute_get_skill_one_all_leads(mjo_ind, fn, rule='Iamp>1.0', month_list=None, datesta=None, dateend=None, lead_max=30, exp_num='1',
                       Fnmjo = '/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/target/romi/ROMI_NOAA_1979to2022.nc'):    
    bcc, rmse = get_skill_one_all_leads(mjo_ind, fn, rule=rule, month_list=month_list, Fnmjo=Fnmjo, datesta=datesta, dateend=dateend, lead_max=lead_max)
    return exp_num, {'bcc': bcc, 'rmse': rmse}

def get_skill_all_leads_parallel(mjo_ind, datesta='2015-01-01', dateend='2018-12-31', fn_list=[], rule='Iamp>1.0', month_list=None, lead_max=30, exp_list=['1',],
                       Fnmjo='/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/target/romi/ROMI_NOAA_1979to2022.nc'):

    bcc_list = {}
    rmse_list = {}

    with ProcessPoolExecutor() as executor:
        # Submit tasks to the executor
        futures = [
            executor.submit(compute_get_skill_one_all_leads, mjo_ind, fn=fn, rule=rule, 
                            month_list=month_list, lead_max=lead_max, exp_num=exp_num, Fnmjo=Fnmjo, datesta=datesta, dateend=dateend)
            for fn in fn_list for exp_num in exp_list
        ]
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            exp_num, result = future.result()
            bcc_list[exp_num] = result['bcc']
            rmse_list[exp_num] = result['rmse']
                
    return bcc_list, rmse_list


def skill_hovallleads_ensemble_mean(
    fn_list = [],
    exp_num_list = np.arange(1,101),
    lat_lim = 10,
    mjo_ind = 'ROMI',
    leadmjo = 35,
    dataflg = 'era5',
    datesta='2016-01-01',
    dateend='2021-12-31',
    ampthred=1, 
    Fnmjo = '/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/target/romi/ROMI_NOAA_1979to2022.nc'
    ):

    rmm = xr.open_dataset(Fnmjo)[mjo_ind].sel(time=slice(datesta, dateend))
    amp = (rmm[:,0]**2 + rmm[:,1]**2)**0.5

    # first calculate the ensemble mean, then calculate bcc and rmse
    dss = [xr.open_dataset(fn).sel(time=slice(datesta, dateend)) for fn in fn_list]
    ds = xr.concat(dss, dim='exp_num')
    
    ds_sel = ds.where(amp>ampthred, drop=True)

    ds_mean = ds_sel.mean(dim='exp_num')

    bcc = []
    rmse = []

    for i in np.arange(0,leadmjo+1):
        bcc_sel = np.sum(ds_mean['predictions'][:,i*2] * ds_mean['targets'][:,i*2] + ds_mean['predictions'][:,2*i+1]*ds_mean['targets'][:,2*i+1]) / (np.sqrt(np.sum(ds_mean['predictions'][:,i*2]**2 + ds_mean['predictions'][:,i*2+1]**2)) * np.sqrt(np.sum(ds_mean['targets'][:,2*i]**2 + ds_mean['targets'][:,2*i+1]**2)))
        bcc.append(bcc_sel)
        rmse_sel = np.sqrt(np.mean((ds_mean['predictions'][:,i*2] - ds_mean['targets'][:,i*2])**2 + (ds_mean['predictions'][:,i*2+1] - ds_mean['targets'][:,i*2+1])**2))
        rmse.append(rmse_sel)

    bcc = np.array(bcc)
    rmse = np.array(rmse)

    return bcc, rmse

    

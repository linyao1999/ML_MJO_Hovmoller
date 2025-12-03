import numpy as np 
import xarray as xr 
import matplotlib.pyplot as plt
import sys
sys.path.append('/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/src/utils')

import WheelerKiladis_util as wk 

# flg = ''
# wgt = False 
flg = 'hann'
wgt = True 

fn = '/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/raw/olr.day.noaa.2x2.nc'
ds = xr.open_dataset(fn, engine='netcdf4')

olr = ds['olr'].sel(time=slice('1980-01-01', '2001-12-31'))

sym, asym = wk.spacetime_power(olr, segsize=96, noverlap=60, spd=1, lat_lim=10, remove_low=True, sigtest=False)
background = wk.power_bag(wk.power_avg(sym, asym))

# each 3 min
# sym2, asym2 = wk.spacetime_power_runningavg(olr, segsize=96, noverlap=60, spd=1, lat_lim=10, remove_low=True, sigtest=False, window_len=2, weighted=wgt)
sym5, asym5 = wk.spacetime_power_runningavg(olr, segsize=96, noverlap=60, spd=1, lat_lim=10, remove_low=True, sigtest=False, window_len=5, weighted=wgt)
sym11, asym11 = wk.spacetime_power_runningavg(olr, segsize=96, noverlap=60, spd=1, lat_lim=10, remove_low=True, sigtest=False, window_len=11, weighted=wgt)
sym21, asym21 = wk.spacetime_power_runningavg(olr, segsize=96, noverlap=60, spd=1, lat_lim=10, remove_low=True, sigtest=False, window_len=21, weighted=wgt)
sym31, asym31 = wk.spacetime_power_runningavg(olr, segsize=96, noverlap=60, spd=1, lat_lim=10, remove_low=True, sigtest=False, window_len=31, weighted=wgt)

# each 3 min
# sym2_minus, asym2_minus = wk.spacetime_power_runningavg_minus(olr, segsize=96, noverlap=60, spd=1, lat_lim=10, remove_low=True, sigtest=False, window_len=2, weighted=wgt)
sym5_minus, asym5_minus = wk.spacetime_power_runningavg_minus(olr, segsize=96, noverlap=60, spd=1, lat_lim=10, remove_low=True, sigtest=False, window_len=5, weighted=wgt)
sym11_minus, asym11_minus = wk.spacetime_power_runningavg_minus(olr, segsize=96, noverlap=60, spd=1, lat_lim=10, remove_low=True, sigtest=False, window_len=11, weighted=wgt)
sym21_minus, asym21_minus = wk.spacetime_power_runningavg_minus(olr, segsize=96, noverlap=60, spd=1, lat_lim=10, remove_low=True, sigtest=False, window_len=21, weighted=wgt)
sym31_minus, asym31_minus = wk.spacetime_power_runningavg_minus(olr, segsize=96, noverlap=60, spd=1, lat_lim=10, remove_low=True, sigtest=False, window_len=31, weighted=wgt)

# store all the variables into nc file
ds_out = xr.Dataset({
    'sym': sym,
    'asym': asym,
    # 'sym2': sym2,
    # 'asym2': asym2,
    'sym5': sym5,
    'asym5': asym5,
    'sym11': sym11,
    'asym11': asym11,   
    'sym21': sym21,
    'asym21': asym21,
    'sym31': sym31,
    'asym31': asym31,
    # 'sym2_minus': sym2_minus,
    # 'asym2_minus': asym2_minus,
    'sym5_minus': sym5_minus,
    'asym5_minus': asym5_minus,
    'sym11_minus': sym11_minus,
    'asym11_minus': asym11_minus,
    'sym21_minus': sym21_minus,
    'asym21_minus': asym21_minus,
    'sym31_minus': sym31_minus,
    'asym31_minus': asym31_minus,
    'background': background,
})

output_fn = f'/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/notebooks/analysis/olr_runningavg{flg}_spectra.nc'
ds_out.to_netcdf(output_fn)

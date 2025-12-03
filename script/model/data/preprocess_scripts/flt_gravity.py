import numpy as np 
import xarray as xr 
import matplotlib.pyplot as plt
# append a path to the sys.path
import sys
sys.path.append('/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/src/utils')
import WheelerKiladis_util as wk 

# salloc --nodes 1 --qos interactive --time 04:00:00 --constraint cpu --account=m3312

flg = 'noaa'
# convert a hovmoller diagram to a wheeler-kiladis diagram
# input_path = '/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/raw/olr.day.mean.nc'
input_path = '/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/raw/olr.day.noaa.2x2.nc'
date_start = '1979-01-01'
date_end = '2022-12-31'
lat_range = 10
olr = xr.open_dataarray(input_path).sel(time=slice(date_start, date_end), lat=slice(lat_range, -lat_range))

# extract the MJO signal
olr_kelvin = wk.filter_olr(olr, kmin=1, kmax=10, fhig=1/3, flow=1/30)
print('olr_kelvin done!')
olr_gravity = wk.filter_olr(olr, kmin=None, kmax=None, fhig=1.0, flow=1/3)
print('olr_gravity done!')


# store olr_mjo into a netcdf file
output_path = f'/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/processed/filtered/olr{flg}.kelvin.k1to10_T3to30.nc'
data = xr.Dataset({'olr': olr_kelvin})
data.to_netcdf(output_path)

print('olr_kelvin output done!')

output_path = f'/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/processed/filtered/olr{flg}.gravity.T2to3.nc'
data = xr.Dataset({'olr': olr_gravity})
data.to_netcdf(output_path)

print('olr_gravity output done!')

power_sym_kelvin = wk.spacetime_power_sym(olr_kelvin)
print('power_sym_kelvin done!')

power_sym_gravity = wk.spacetime_power_sym(olr_gravity)
print('power_sym_gravity done!')

output_path = f'/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/processed/plots/power{flg}_spectra_sym_kelvin_k1to10_T3to30.nc'
data = xr.Dataset({'power_sym_kelvin': power_sym_kelvin})
data.to_netcdf(output_path)

output_path = f'/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/processed/plots/power{flg}_spectra_sym_gravity_T2to3.nc'
data = xr.Dataset({'power_sym_gravity': power_sym_gravity})
data.to_netcdf(output_path)

print('spectra output done!')   

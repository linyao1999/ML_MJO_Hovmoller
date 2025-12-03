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

power_sym = wk.spacetime_power_sym(olr)
print('power_sym done!')


# extract the MJO signal
olr_mjo = wk.filter_olr(olr, kmin=1, kmax=5, fhig=1/20, flow=1/100)
print('olr_mjo done!')
olr_kelvin = wk.filter_olr(olr, kmin=1, kmax=10, fhig=1/2.5, flow=1/30)
print('olr_kelvin done!')
olr_rossby = wk.filter_olr(olr, kmin=-10, kmax=-1, fhig=1/10, flow=1/100)
print('olr_rossby done!')


# store olr_mjo into a netcdf file
output_path = f'/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/processed/filtered/olr{flg}.mjo.k1to5_T20to100.nc'
data = xr.Dataset({'olr': olr_mjo})
data.to_netcdf(output_path)

print('olr_mjo output done!')

output_path = f'/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/processed/filtered/olr{flg}.kelvin.k1to10_T2.5to30.nc'
data = xr.Dataset({'olr': olr_kelvin})
data.to_netcdf(output_path)

print('olr_kelvin output done!')

output_path = f'/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/processed/filtered/olr{flg}.rossby.k-1to-10_T10to100.nc'
data = xr.Dataset({'olr': olr_rossby})
data.to_netcdf(output_path)

print('olr_rossby output done!')

power_sym_mjo = wk.spacetime_power_sym(olr_mjo)
print('power_sym_mjo done!')

power_sym_kelvin = wk.spacetime_power_sym(olr_kelvin)
print('power_sym_kelvin done!')

power_sym_rossby = wk.spacetime_power_sym(olr_rossby)
print('power_sym_rossby done!')

# store power_sym and power_sym_mjo into a netcdf file
output_path = f'/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/processed/plots/power{flg}_spectra_sym_rawandmjo_k1to5_T20to100.nc'
data = xr.Dataset({'power_sym': power_sym, 'power_sym_mjo': power_sym_mjo})
data.to_netcdf(output_path)

output_path = f'/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/processed/plots/power{flg}_spectra_sym_kelvin_k1to10_T2.5to30.nc'
data = xr.Dataset({'power_sym_kelvin': power_sym_kelvin})
data.to_netcdf(output_path)

output_path = f'/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/processed/plots/power{flg}_spectra_sym_rossby_k-1to-10_T10to100.nc'
data = xr.Dataset({'power_sym_rossby': power_sym_rossby})
data.to_netcdf(output_path)

print('spectra output done!')   

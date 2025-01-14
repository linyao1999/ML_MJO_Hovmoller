import numpy as np 
import pandas as pd 
import xarray as xr 
import matplotlib.pyplot as plt 
import os 
import logging
import shutil
from datetime import datetime
from scipy.signal import detrend
from scipy.signal import convolve2d
from scipy.stats import ttest_1samp
from scipy.stats import linregress
from scipy import special
import math
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import dask.array as da

# Define the colormap from the 'coolwarm' base colormap
original_coolwarm = plt.cm.get_cmap('coolwarm')

# Levels for the colorbar
levels = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.7, 2, 2.4, 2.8, 3.2, 3.6, 4]

# Create a BoundaryNorm object for mapping data values to colors
norm = mcolors.BoundaryNorm(levels, original_coolwarm.N, clip=False)


# calculate the Wheeler and Kiladis plot (wavenumber v.s. frequency) for input maps
# reference: https://journals.ametsoc.org/view/journals/atsc/56/3/1520-0469_1999_056_0374_ccewao_2.0.co_2.xml

import cftime

# # Function to convert cftime.DatetimeNoLeap to day of year
# def cftime_to_doy(cf_datetime_obj):
#     start_of_year = cftime.DatetimeNoLeap(cf_datetime_obj.year, 1, 1)
#     doy = (cf_datetime_obj - start_of_year).days + 1
#     return doy

def rmv_lowfreq(data):
    # remove the first three harmonics in the input data
    # data[time, lat, lon]
    # this is designed for E3SM-MMF data output. The time format is cftime object
    arr = data.copy()
    doy_values = arr['time'].dt.dayofyear
    arr['doy'] = doy_values

    # annual cycle [doy, lat, lon]
    arr_anu = arr.groupby('doy').mean(dim='time')

    # fft to get smoothed annual cycle
    doy_index = arr_anu.dims.index('doy')
    x_fft = np.fft.rfft(arr_anu.values, axis=doy_index)
    x_fft[4:] = 0.0  # we already know the first dimension is time
    x_re = np.fft.irfft(x_fft, arr_anu.shape[0], axis=doy_index)

    # give the smoothed annual values to a dataarray
    arr_anu.values = x_re  # [doy, lat, lon]

    # remove the first three harmonics from raw data
    out = arr.groupby('doy') - arr_anu

    return out

# decompose the input map into symmetric and antisymmetric parts.
def decompose2SymAsym(arr):
    # copy from https://github.com/brianpm/wavenumber_frequency/blob/master/wavenumber_frequency_functions.py
    """Mimic NCL function to decompose into symmetric and asymmetric parts.
    arr: xarra DataArray
    return: symmetric and asymmetric parts. 
    """
    lat_dim = arr.dims.index('lat')
    # print('decompose along axis=', str(lat_dim))
    data_sym = 0.5*(arr.values + np.flip(arr.values, axis=lat_dim))
    data_asy = 0.5*(arr.values - np.flip(arr.values, axis=lat_dim))
    data_sym = xr.DataArray(data_sym, dims=arr.dims, coords=arr.coords, name='sym')
    data_asy = xr.DataArray(data_asy, dims=arr.dims, coords=arr.coords, name='Asym')

    # # Assuming lat is a coordinate in your xarray Dataset or DataArray
    # latitude_values = arr.lat.values

    # # Open a text file and write latitude values to it
    # with open('latitude_values.txt', 'w') as f:
    #     for value in latitude_values:
    #         f.write(f"{value}\n")

    out = arr.copy()  # might not be best to copy, but is safe        
    out.loc[{'lat':arr['lat'][arr['lat']<0]}] = data_sym.isel(lat=data_sym.lat<0)
    out.loc[{'lat':arr['lat'][arr['lat']>0]}] = data_asy.isel(lat=data_asy.lat>0)
    return out

def split_hann_taper(seg_size, fraction):
    '''
    seg_size: the size of the taper;
    fraction: the fraction of the total points used to do hanning window
    '''
    npts = int(np.rint(seg_size * fraction))  # the total length of hanning window
    hann_taper = np.hanning(npts)  # generate Hanning taper
    taper = np.ones(seg_size)  # create the split cosine bell taper

    # copy the first half of hanner taper to target taper
    taper[:npts//2] = hann_taper[:npts//2]
    # copy the second half of hanner taper to the target taper
    # taper[-npts//2-1:] = hann_taper[-npts//2-1:]
    taper[-(npts//2):] = hann_taper[-(npts//2):]
    return taper

def split_hann_taper_pnt(seg_size, pnt=100):
    '''
    seg_size: the size of the taper;
    pnt: the number of the total points used to do hanning window
    '''
    npts = int(pnt)  # the total length of hanning window
    hann_taper = np.hanning(npts)  # generate Hanning taper
    taper = np.ones(seg_size)  # create the split cosine bell taper

    # copy the first half of hanner taper to target taper
    taper[:npts//2] = hann_taper[:npts//2]
    # copy the second half of hanner taper to the target taper
    # taper[-npts//2-1:] = hann_taper[-npts//2-1:]
    taper[-(npts//2):] = hann_taper[-(npts//2):]
    return taper

def Hayashi(varfft, nday):
    # use Hayashi method to reorder wavenumber-frequency matrix
    # For ffts that return the coefficients as described above, here is the algorithm
    # coeff array varfft(...,n,t)   dimensioned (...,0:numlon-1,0:numtim-1)
    # new space/time pee(...,pn,pt) dimensioned (...,0:numlon  ,0:numtim  ) 
    #
    # NOTE: one larger in both freq/space dims
    # copied from ncl script
    #
    #    if  |  0 <= pn <= numlon/2-1    then    | numlon/2 <= n <= 1
    #        |  0 <= pt < numtim/2-1             | numtim/2 <= t <= numtim-1
    #
    #    if  |  0         <= pn <= numlon/2-1    then    | numlon/2 <= n <= 1
    #        |  numtime/2 <= pt <= numtim                | 0        <= t <= numtim/2
    #
    #    if  |  numlon/2  <= pn <= numlon    then    | 0  <= n <= numlon/2
    #        |  0         <= pt <= numtim/2          | numtim/2 <= t <= 0
    #
    #    if  |  numlon/2   <= pn <= numlon    then    | 0        <= n <= numlon/2
    #        |  numtim/2+1 <= pt <= numtim            | numtim-1 <= t <= numtim/2
    mlon = len(varfft['wavenumber'])
    mtim = len(varfft['frequency'])

    M = ((mlon - 1)//2) * 2 + 1  # the odd number <= mlon
    N = ((mtim - 1)//2) * 2 + 1  # the odd number <= mtim

    varspacetime = np.empty((varfft.shape[0],varfft.shape[1], M, N), dtype=varfft.dtype)

    varspacetime[:, :, 0:((mlon-1)//2), 0:((mtim-1)//2) ] = varfft[:, :, ((mlon-1)//2):0:-1, -((mtim-1)//2): ]  
    varspacetime[:, :, 0:((mlon-1)//2), ((mtim-1)//2): ] = varfft[:, :, ((mlon-1)//2):0:-1, 0:((mtim-1)//2 + 1) ]  
    varspacetime[:, :, ((mlon-1)//2):, 0:((mtim-1)//2 + 1) ] = varfft[:, :, 0:((mlon-1)//2 + 1), ((mtim-1)//2)::-1 ]  
    varspacetime[:, :, ((mlon-1)//2):, ((mtim-1)//2 + 1): ] = varfft[:, :, 0:((mlon-1)//2 + 1), -1:(-((mtim-1)//2) - 1):-1]  

    # print('test')
    pee = np.absolute(varspacetime)**2
    # print('test1')
    wave = np.arange(-((mlon - 1)//2), ((mlon - 1)//2 + 1), 1, dtype=int)
    freq = np.linspace((- ((mtim-1)//2)/nday), ((mtim-1)//2)/nday, N)
    
    out = xr.DataArray(
        data=pee,
        dims=("time","lat","wavenumber","frequency"),
        coords={
            "time": varfft["time"],
            "lat": varfft["lat"],
            "wavenumber": wave,
            "frequency": freq,
        }
    )

    return out

def spacetime_power(data, segsize=96, noverlap=60, spd=1, lat_lim=15, remove_low=True, sigtest=False):
    """
    Perform space-time spectral decomposition and return raw power spectrum following Wheeler-Kiladis approach.

    data: an xarray DataArray to be analyzed; needs to have (time, lat, lon) dimensions.
    segsize: integer (days) denoting the size of time samples that will be decomposed (typically about 96)
    noverlap: integer (days) denoting the number of days of overlap from one segment to the next
    spd: sampling rate, in "samples per day" (e.g. daily=1, 6-houry=4)
    latitude_bounds: a tuple of (southern_extent, northern_extent) to reduce data size.

    Method
    ------------------
        0. Remove the first three harmonics of annual cycle to prevent aliasing.
        1. Subsample in latitude if latitude_bounds is specified.
        2. Construct symmetric/antisymmetric array .
        3. Construct overlapping window view of data.
        4. Detrend the segments (remove linear trend).
        5. Apply taper in time dimension of windows (aka segments).
        6. Fourier transform
        7. calculate the power.
        8. average over all segments
        9. sum the power over all latitudes.
        
    Notes
    ---------------------------
        Upon returning power, this should be comparable to "raw" spectra. 
        Next step would be be to smooth with `smooth_wavefreq`, 
        and divide raw spectra by smooth background to obtain "significant" spectral power.
        
    """

    segsize = spd * segsize  # how many time steps included.
    noverlap = spd * noverlap  

    # select the interested section in data
    # # NOTE: starting from negative values; if your dataset starts from positive latitudes, please revise the following line
    # data1 = data.sel(lat=slice(-lat_lim, lat_lim))
    # NOTE: starting from positive values; if your dataset starts from negative latitudes, please revise the following line
    data1 = data.sel(lat=slice(lat_lim, -lat_lim))
    # # Assuming lat is a coordinate in your xarray Dataset or DataArray
    # latitude_values = data1.lat.values

    # # Open a text file and write latitude values to it
    # with open('latitude_values_data1.txt', 'w') as f:
    #     for value in latitude_values:
    #         f.write(f"{value}\n")

    # 0. remove low-frequency signals
    if remove_low:
        data2 = rmv_lowfreq(data1)
    else:
        data2 = data1

    # 2. [time, lat(pos+neg), lon]
    # lat<0: symmetric; lat>0: antisymmetric
    data_sym_asym = decompose2SymAsym(data2)

    # 3. 
    data_sym_asym0 = data_sym_asym.rolling(time=segsize, min_periods=segsize).construct("segments").dropna('time')  # WK99 use 96-day window
    # x_roll_asym0 = data_asym.rolling(time=segsize, min_periods=segsize).construct("segments").dropna('time')  # WK99 use 96-day window
    
    # set overlaps
    x_roll_sym_asym = data_sym_asym0.isel(time=slice(None, None, segsize-noverlap))
    # x_roll_asym = x_roll_asym0.isel(time=slice(None, None, segsize-noverlap))

    seg_dim = x_roll_sym_asym.dims.index('segments')
    # print('seg_dim: ' + str(seg_dim))

    # 4. Detrend the linear trend
    # # Apply the detrend function to each segment
    # x_detrend_sym_asym = xr.apply_ufunc(
    #     detrend,
    #     x_roll_sym_asym,
    #     kwargs={'axis': seg_dim},
    #     dask='parallelized',
    #     output_dtypes=[x_roll_sym_asym.dtype]
    # )

    # chunk the data to avoid memory error

    # print('coordinates: ', x_roll_sym_asym.dims)
    # print('size of x_roll_sym_asym: ', x_roll_sym_asym.shape)

    original_dims = x_roll_sym_asym.dims
    x_roll_sym_asym = x_roll_sym_asym.chunk({'time': 1, 'lat': 'auto', 'lon': 'auto', 'segments': -1})

    print('coordinates: ', x_roll_sym_asym.dims)
    print('size of x_roll_sym_asym: ', x_roll_sym_asym.shape)

    def detrend_timeseries(ts):
        time_nums = np.arange(ts.size)
        coeffs = np.polyfit(time_nums, ts, 1)
        trend = np.polyval(coeffs, time_nums)
        return ts - trend

    # Use apply_ufunc to apply the detrending
    detrended_data = xr.apply_ufunc(
        detrend_timeseries,
        x_roll_sym_asym,  # Your Dask-backed DataArray
        input_core_dims=[['segments']],
        output_core_dims=[['segments']],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[x_roll_sym_asym.dtype]
    )

    # print('detrended_data: ', detrended_data.shape)
    x_detrend_sym_asym = detrended_data.compute()

    # x_detrend_sym_asym = x_detrend_sym_asym.transpose(*original_dims)

    # print('coordinates: ', x_detrend_sym_asym.dims)
    # print('size of x_roll_sym_asym: ', x_detrend_sym_asym.shape)

    # 5. add taper
    # using hanning window w[n] = 0.5 * (1 - cos(2*pi*n/(M-1))) to create split cosine bell taper
    taper = split_hann_taper(seg_size=segsize, fraction=0.5)

    # print('size of taper: ', np.shape(taper))

    x_detrend_tap = x_detrend_sym_asym * taper  # time, lat, lon, segments
    # x_detrend_asymtap = x_detrend_asym * taper

    # 6. fourier transform within 2 steps 
    # perform fft in longitude
    lon_dim = x_detrend_tap.dims.index('lon')
    lon_size = x_detrend_tap.shape[lon_dim]
    # following ncl script, we normalize the fft coefficients with lon_size
    fft_lon = np.fft.fft(x_detrend_tap, axis=lon_dim) / lon_size # time, lat, wavenumber, segments
    # fft_lon_asym = np.fft.fft(x_detrend_asymtap, axis=lon_dim) / lon_size # time, lat, wavenumber, segments
    # perform fft in segments
    seg_dim = x_detrend_tap.dims.index('segments')
    seg_size = x_detrend_tap.shape[seg_dim]
    fft_lonseg = np.fft.fft(fft_lon, axis=seg_dim) / seg_size  # time, lat, wavenumber, frequency
    # fft_lonseg_asym = np.fft.fft(fft_lon_asym, axis=seg_dim) / seg_size  # time, lat, wavenumber, frequency

    fft_sym_asym = xr.DataArray(
        data=fft_lonseg,
        dims=("time","lat","wavenumber","frequency"),
        coords={
            "time": x_detrend_tap["time"],
            "lat": x_detrend_tap["lat"],
            "wavenumber": np.fft.fftfreq(lon_size, 1/lon_size),  # how many cycles along the equator 
            "frequency": np.fft.fftfreq(seg_size, 1/spd),  # the frequency (day-1)
        }
    )
    # fft_asym = xr.DataArray(
    #     data=fft_lonseg_asym,
    #     dims=("time","lat","wavenumber","frequency"),
    #     coords={
    #         "time": x_detrend_symtap["time"],
    #         "lat": x_detrend_symtap["lat"],
    #         "wavenumber": np.fft.fftfreq(lon_size, 1/lon_size),  # how many cycles along the equator 
    #         "frequency": np.fft.fftfreq(seg_size, 1/spd),  # the frequency (day-1)
    #     }
    # )

    # 7. [time, lat, wavenumber, frequency]
    # reorder coef matrix according to ncl script
    fft_reorder = Hayashi(fft_sym_asym, segsize/spd)
    # fft_asym_reorder = Hayashi(fft_asym, segsize/spd)
    
    # 8. average over all segments [wavenumber, frequency]
    zsym = 2.0 * fft_reorder.isel(lat=fft_reorder.lat<0).mean(dim='time').sum(dim='lat').squeeze()
    zsym.name = "power"
    zasym = 2.0 * fft_reorder.isel(lat=fft_reorder.lat>0).mean(dim='time').sum(dim='lat').squeeze()
    zasym.name = "power"

    if sigtest:
        # get power spectra for each segment for significance test [time, wavenumber, frequency]
        zsym1 = 2.0 * fft_reorder.isel(lat=fft_reorder.lat<0).sum(dim='lat').squeeze()
        zasym1 = 2.0 * fft_reorder.isel(lat=fft_reorder.lat>0).sum(dim='lat').squeeze()
        return zsym.where(zsym['frequency']>0, drop=True), zasym.where(zasym['frequency']>0, drop=True), zsym1.where(zsym1['frequency']>0, drop=True), zasym1.where(zasym1['frequency']>0, drop=True)

    else:
        return zsym.where(zsym['frequency']>0, drop=True), zasym.where(zasym['frequency']>0, drop=True)

def spacetime_power_sym(data, segsize=96, noverlap=60, spd=1, lat_lim=15, remove_low=True):
    """
    Perform space-time spectral decomposition and return raw power spectrum following Wheeler-Kiladis approach.

    data: an xarray DataArray to be analyzed; needs to have (time, lat, lon) dimensions.
    segsize: integer (days) denoting the size of time samples that will be decomposed (typically about 96)
    noverlap: integer (days) denoting the number of days of overlap from one segment to the next
    spd: sampling rate, in "samples per day" (e.g. daily=1, 6-houry=4)
    latitude_bounds: a tuple of (southern_extent, northern_extent) to reduce data size.

    Method
    ------------------
        0. Remove the first three harmonics of annual cycle to prevent aliasing.
        1. Subsample in latitude if latitude_bounds is specified.
        2. Construct symmetric/antisymmetric array .
        3. Construct overlapping window view of data.
        4. Detrend the segments (remove linear trend).
        5. Apply taper in time dimension of windows (aka segments).
        6. Fourier transform
        7. calculate the power.
        8. average over all segments
        9. sum the power over all latitudes.
        
    Notes
    ---------------------------
        Upon returning power, this should be comparable to "raw" spectra. 
        Next step would be be to smooth with `smooth_wavefreq`, 
        and divide raw spectra by smooth background to obtain "significant" spectral power.
        
    """

    segsize = spd * segsize  # how many time steps included.
    noverlap = spd * noverlap  

    # select the interested section in data
    # # NOTE: starting from negative values; if your dataset starts from positive latitudes, please revise the following line
    # data1 = data.sel(lat=slice(-lat_lim, lat_lim)).load()
    # NOTE: starting from positive values; if your dataset starts from negative latitudes, please revise the following line
    data1 = data.sel(lat=slice(lat_lim, -lat_lim)).load()
    # # Assuming lat is a coordinate in your xarray Dataset or DataArray
    # latitude_values = data1.lat.values

    # # Open a text file and write latitude values to it
    # with open('latitude_values_data1.txt', 'w') as f:
    #     for value in latitude_values:
    #         f.write(f"{value}\n")

    # 0. remove low-frequency signals
    if remove_low:
        data2 = rmv_lowfreq(data1)
    else:
        data2 = data1

    # 2. [time, lat(pos+neg), lon]
    data_sym_asym = decompose2SymAsym(data2)

    # 3. 
    data_sym_asym0 = data_sym_asym.rolling(time=segsize, min_periods=segsize).construct("segments").dropna('time')  # WK99 use 96-day window
    # x_roll_asym0 = data_asym.rolling(time=segsize, min_periods=segsize).construct("segments").dropna('time')  # WK99 use 96-day window
    
    # set overlaps
    x_roll_sym_asym = data_sym_asym0.isel(time=slice(None, None, segsize-noverlap))
    # x_roll_asym = x_roll_asym0.isel(time=slice(None, None, segsize-noverlap))

    seg_dim = x_roll_sym_asym.dims.index('segments')
    # print('seg_dim: ' + str(seg_dim))

    # # 4. 
    # # Apply the detrend function to each segment

    original_dims = x_roll_sym_asym.dims
    x_roll_sym_asym = x_roll_sym_asym.chunk({'time': 1, 'lat': 'auto', 'lon': 'auto', 'segments': -1})

    print('coordinates: ', x_roll_sym_asym.dims)
    print('size of x_roll_sym_asym: ', x_roll_sym_asym.shape)

    def detrend_timeseries(ts):
        time_nums = np.arange(ts.size)
        coeffs = np.polyfit(time_nums, ts, 1)
        trend = np.polyval(coeffs, time_nums)
        return ts - trend

    # Use apply_ufunc to apply the detrending
    detrended_data = xr.apply_ufunc(
        detrend_timeseries,
        x_roll_sym_asym,  # Your Dask-backed DataArray
        input_core_dims=[['segments']],
        output_core_dims=[['segments']],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[x_roll_sym_asym.dtype]
    )

    # print('detrended_data: ', detrended_data.shape)
    x_detrend_sym_asym = detrended_data.compute()


    # 5. add taper
    # using hanning window w[n] = 0.5 * (1 - cos(2*pi*n/(M-1))) to create split cosine bell taper
    taper = split_hann_taper(seg_size=segsize, fraction=0.5)

    print('size of taper: ', np.shape(taper))

    x_detrend_tap = x_detrend_sym_asym * taper  # time, lat, lon, segments
    # x_detrend_asymtap = x_detrend_asym * taper

    # 6. fourier transform within 2 steps 
    # perform fft in longitude
    lon_dim = x_detrend_tap.dims.index('lon')
    lon_size = x_detrend_tap.shape[lon_dim]
    # following ncl script, we normalize the fft coefficients with lon_size
    fft_lon = np.fft.fft(x_detrend_tap, axis=lon_dim) / lon_size # time, lat, wavenumber, segments
    # fft_lon_asym = np.fft.fft(x_detrend_asymtap, axis=lon_dim) / lon_size # time, lat, wavenumber, segments
    # perform fft in segments
    seg_dim = x_detrend_tap.dims.index('segments')
    seg_size = x_detrend_tap.shape[seg_dim]
    fft_lonseg = np.fft.fft(fft_lon, axis=seg_dim) / seg_size  # time, lat, wavenumber, frequency
    # fft_lonseg_asym = np.fft.fft(fft_lon_asym, axis=seg_dim) / seg_size  # time, lat, wavenumber, frequency

    fft_sym_asym = xr.DataArray(
        data=fft_lonseg,
        dims=("time","lat","wavenumber","frequency"),
        coords={
            "time": x_detrend_tap["time"],
            "lat": x_detrend_tap["lat"],
            "wavenumber": np.fft.fftfreq(lon_size, 1/lon_size),  # how many cycles along the equator 
            "frequency": np.fft.fftfreq(seg_size, 1/spd),  # the frequency (day-1)
        }
    )
    # fft_asym = xr.DataArray(
    #     data=fft_lonseg_asym,
    #     dims=("time","lat","wavenumber","frequency"),
    #     coords={
    #         "time": x_detrend_symtap["time"],
    #         "lat": x_detrend_symtap["lat"],
    #         "wavenumber": np.fft.fftfreq(lon_size, 1/lon_size),  # how many cycles along the equator 
    #         "frequency": np.fft.fftfreq(seg_size, 1/spd),  # the frequency (day-1)
    #     }
    # )

    # 7. [time, lat, wavenumber, frequency]
    # reorder coef matrix according to ncl script
    fft_reorder = Hayashi(fft_sym_asym, segsize/spd)  # k is from -wavenumber to wavenumber; f is from -frequency to frequency
    # fft_asym_reorder = Hayashi(fft_asym, segsize/spd)
    
    # 8. average over all segments [wavenumber, frequency]
    zsym = 2.0 * fft_reorder.isel(lat=fft_reorder.lat<0).mean(dim='time').sum(dim='lat').squeeze()
    zsym.name = "power"

    return zsym.where(zsym['frequency']>0, drop=True)

def power_avg(sym, asym):
    # average over symmetric and antisymmetric components
    return (sym + asym) * 0.5

def wk_smooth121(x):
    # x is a 1-d numpy array
    # create running average kernal
    kern = np.asarray([1/4,1/2,1/4])
    x1 = np.concatenate(([x[0]], x, [x[-1]]))
    x2 = np.convolve(x1, kern, mode='valid')
    return x2

def power_bag(zavg):
    '''
    zavg: averaged power spectrum [wavenuber, frequency]
    1. smooth wavenumbers
    fq < 0.1, smooth 5 times
    fq < 0.2, smooth 10 times
    fq < 0.3, smooth 20 times
    fq >=0.3, smooth 40 times

    2. smooth positive frequency up to 0.8 cpd (max) 10 times
    '''

    x = zavg.where(zavg['frequency']>0, drop=True) # [wavenumber, positive freq]
    
    fq = x['frequency']
    wn = x['wavenumber']

    # Smooth wavenumbers
    smoothed_x = np.copy(x.values)

    for i in range(len(fq)):
        if fq[i] < 0.1:
            for _ in range(5):
                smoothed_x[:, i] = wk_smooth121(smoothed_x[:, i])
        elif fq[i] < 0.2:
            for _ in range(10):
                smoothed_x[:, i] = wk_smooth121(smoothed_x[:, i])
        elif fq[i] < 0.3:
            for _ in range(20):
                smoothed_x[:, i] = wk_smooth121(smoothed_x[:, i])
        elif fq[i] >= 0.3:
            for _ in range(40):
                smoothed_x[:, i] = wk_smooth121(smoothed_x[:, i])

    # Smooth positive frequency up to 0.8 cpd (max) 10 times
    # pt8cpd = min(np.where(fq >= 0.8)[0])
    
    for i in range(len(wn)):
        for _ in range(10):
            smoothed_x[i, :] = wk_smooth121(smoothed_x[i, :])

    x.values = smoothed_x

    return x

def wk_analysis(x, **kwargs):
    '''
    x is the data to do WK analysis
    optional kwargs:
    segsize, noverlap, spd, lat_lim, remove_low
    '''

    # get the raw space-time power spectra for symmetric and anti-symmetric components
    # negative frequency has been removed. 
    sym, asym, sym_segs, asym_segs = spacetime_power(x, **kwargs)  
    # sym and asym: [wavenumber, frequency(positive)]
    # sym_segs and asym_segs: [time, wavenumber, frequency(positive)]

    # average between symmteric and anti-symmetric components for background calculation
    zavg = power_avg(sym, asym)

    # following the ncl scripts, we smooth raw power spectra a little bit
    smooth_sym = sym.copy()
    smooth_asym = asym.copy()
    # smooth along frequency
    wn = sym['wavenumber']
    for i in range(len(wn)):
        smooth_sym[i, :].values = wk_smooth121(sym[i,:].values)
        smooth_asym[i, :].values = wk_smooth121(asym[i,:].values)

    # # remove spurious power (frequency=0)
    # sym.loc[{'frequency':0}] = np.nan
    # asym.loc[{'frequency':0}] = np.nan

    # get the background based on the component average zavg
    background = power_bag(zavg)    

    # normalize using background
    sym_norm = smooth_sym / background
    asym_norm = smooth_asym / background

    # test the significance
    # H0: sym_sges.mean('time') = background
    # sym_sges [time, wavenumber, frequency]
    # background [wavenumber, frequency]

    # sample mean: sym, asym
    # population mean: background
    # # sample standard deviation: 
    # sym_std = sym_segs.std(dim='time').squeeze()
    # asym_std = asym_segs.std(dim='time').squeeze()
    # # number of observations:
    # n = len(sym_segs['time'])

    # sym_t_score = (sym - background) / sym_std
    # calculate the t score
    print(sym_segs.shape)
    print(background.shape)

    tscore_sym, pvalue_sym = ttest_1samp(sym_segs.values, np.reshape(background.values, (1, background.shape[0], background.shape[1])), axis=0, alternative='greater')
    p_sym = np.copy(pvalue_sym)
    p_sym[pvalue_sym < 0.01] = 1.0 # if pvalue is < 0.01, we reject H0 and accept alternative hypothesis.
    p_sym[pvalue_sym >= 0.01] = np.nan

    tscore_asym, pvalue_asym = ttest_1samp(asym_segs.values, np.reshape(background.values, (1, background.shape[0], background.shape[1])), axis=0, alternative='greater')
    p_asym = np.copy(pvalue_asym)
    p_asym[pvalue_asym < 0.01] = 1.0 # if pvalue is < 0.01, we reject H0 and accept alternative hypothesis.
    p_asym[pvalue_asym >= 0.01] = np.nan

    return smooth_sym, smooth_asym, background, sym_norm, asym_norm, sym_norm*p_sym, asym_norm*p_asym

def genDispersionCurves(nWaveType=6, nPlanetaryWave=50, rlat=0, Ahe=[50, 20, 10]):
    """
    Function to derive the shallow water dispersion curves. Closely follows NCL version.

    input:
        nWaveType : integer, number of wave types to do
        nPlanetaryWave: integer
        rlat: latitude in radians (just one latitude, usually 0.0)
        Ahe: [50.,25.,12.] equivalent depths
              ==> defines parameter: nEquivDepth ; integer, number of equivalent depths to do == len(Ahe)

    returns: tuple of size 2
        Afreq: Frequency, shape is (nWaveType, nEquivDepth, nPlanetaryWave)
        Apzwn: Zonal savenumber, shape is (nWaveType, nEquivDepth, nPlanetaryWave)
        
    notes:
        The outputs contain both symmetric and antisymmetric waves. In the case of 
        nWaveType == 6:
        0,1,2 are (ASYMMETRIC) "MRG", "IG", "EIG" (mixed rossby gravity, inertial gravity, equatorial inertial gravity)
        3,4,5 are (SYMMETRIC) "Kelvin", "ER", "IG" (Kelvin, equatorial rossby, inertial gravity)
    """
    nEquivDepth = len(Ahe) # this was an input originally, but I don't know why.
    pi    = np.pi
    radius = 6.37122e06    # [m]   average radius of earth
    g     = 9.80665        # [m/s] gravity at 45 deg lat used by the WMO
    omega = 7.292e-05      # [1/s] earth's angular vel
    # U     = 0.0   # NOT USED, so Commented
    # Un    = 0.0   # since Un = U*T/L  # NOT USED, so Commented
    ll    = 2.*pi*radius*np.cos(np.abs(rlat))
    Beta  = 2.*omega*np.cos(np.abs(rlat))/radius
    fillval = 1e20
    
    # NOTE: original code used a variable called del,
    #       I just replace that with `dell` because `del` is a python keyword.

    # Initialize the output arrays
    Afreq = np.empty((nWaveType, nEquivDepth, nPlanetaryWave))
    Apzwn = np.empty((nWaveType, nEquivDepth, nPlanetaryWave))

    for ww in range(1, nWaveType+1):
        for ed, he in enumerate(Ahe):
            # this loops through the specified equivalent depths
            # ed provides index to fill in output array, while
            # he is the current equivalent depth
            # T = 1./np.sqrt(Beta)*(g*he)**(0.25) This is close to pre-factor of the dispersion relation, but is not used.
            c = np.sqrt(g * he)  # phase speed   
            L = np.sqrt(c/Beta)  # was: (g*he)**(0.25)/np.sqrt(Beta), this is Rossby radius of deformation        

            for wn in range(1, nPlanetaryWave+1):
                s  = -20.*(wn-1)*2./(nPlanetaryWave-1) + 20.
                k  = 2.0 * pi * s / ll
                kn = k * L 

                # Anti-symmetric curves  
                if (ww == 1):       # MRG wave
                    if (k < 0):
                        dell  = np.sqrt(1.0 + (4.0 * Beta)/(k**2 * c))
                        deif = k * c * (0.5 - 0.5 * dell)
                    
                    if (k == 0):
                        deif = np.sqrt(c * Beta)
                    
                    if (k > 0):
                        deif = fillval
                    
                
                if (ww == 2):       # n=0 IG wave
                    if (k < 0):
                        deif = fillval
                    
                    if (k == 0):
                        deif = np.sqrt( c * Beta)
                    
                    if (k > 0):
                        dell  = np.sqrt(1.+(4.0*Beta)/(k**2 * c))
                        deif = k * c * (0.5 + 0.5 * dell)
                    
                
                if (ww == 3):       # n=2 IG wave
                    n=2.
                    dell  = (Beta*c)
                    deif = np.sqrt((2.*n+1.)*dell + (g*he) * k**2)
                    # do some corrections to the above calculated frequency.......
                    for i in range(1,5+1):
                        deif = np.sqrt((2.*n+1.)*dell + (g*he) * k**2 + g*he*Beta*k/deif)
                    
    
                # symmetric curves
                if (ww == 4):       # n=1 ER wave
                    n=1.
                    if (k < 0.0):
                        dell  = (Beta/c)*(2.*n+1.)
                        deif = -Beta*k/(k**2 + dell)
                    else:
                        deif = fillval
                    
                if (ww == 5):       # Kelvin wave
                    deif = k*c

                if (ww == 6):       # n=1 IG wave
                    n=1.
                    dell  = (Beta*c)
                    deif = np.sqrt((2. * n+1.) * dell + (g*he)*k**2)
                    # do some corrections to the above calculated frequency
                    for i in range(1,5+1):
                        deif = np.sqrt((2.*n+1.)*dell + (g*he)*k**2 + g*he*Beta*k/deif)
                
                eif  = deif  # + k*U since  U=0.0
                P    = 2.*pi/(eif*24.*60.*60.)  #  => PERIOD
                # dps  = deif/k  # Does not seem to be used.
                # R    = L #<-- this seemed unnecessary, I just changed R to L in Rdeg
                # Rdeg = (180.*L)/(pi*6.37e6) # And it doesn't get used.
            
                Apzwn[ww-1,ed-1,wn-1] = s
                if (deif != fillval):
                    # P = 2.*pi/(eif*24.*60.*60.) # not sure why we would re-calculate now
                    Afreq[ww-1,ed-1,wn-1] = 1./P
                else:
                    Afreq[ww-1,ed-1,wn-1] = fillval
    return  Afreq, Apzwn

def wk_plot_sym(sym, tlt='', logflg=True, savflg=False, pltDispCurve=True, fb=[0,0.48], center0=False, filename='wk.png', setcolor=False, vmax=None, vmin=None, cmapflg='Blues'):
    wavenumber = sym['wavenumber']
    frequency = sym['frequency']

    ftsize = 26

    plt.rcParams.update({'font.size': ftsize})

    # Create a figure and subplots
    fig = plt.figure(figsize=(7,7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.05])

    # Plot smooth_sym for each frequency
    ax = fig.add_subplot(gs[0, 0])
    # ax = fig.add_subplot()
    if logflg:
        if center0:
            vmax = np.max(np.abs(np.log10(sym.T)))
            c = ax.contourf(wavenumber, frequency, np.log10(sym.T), cmap='RdBu_r', levels=15, vmin=-vmax, vmax=vmax)
        else:
            if setcolor:
                c = ax.contourf(wavenumber, frequency, np.log10(sym.T), cmap=cmapflg, vmin=vmin, vmax=vmax, levels=8)
            else:
                c = ax.contourf(wavenumber, frequency, np.log10(sym.T), cmap=cmapflg, levels=8)
    else:
        if setcolor:
            c = ax.contourf(wavenumber, frequency, sym.T, vmin=vmin, vmax=vmax, levels=8, cmap=cmapflg)
        else:
            c = ax.contourf(wavenumber, frequency, sym.T, levels=8, cmap=cmapflg)
        # c = ax.contourf(wavenumber, frequency, sym.T, cmap='Reds', levels=25)

    if pltDispCurve:
        wavfreq, wavwn = genDispersionCurves()
        swf = np.where(wavfreq == 1e20, np.nan, wavfreq)
        swk = np.where(wavwn == 1e20, np.nan, wavwn)

        for ii in range(3,6):
            ax.plot(swk[ii, 0,:], swf[ii,0,:], color='darkgray')
            ax.plot(swk[ii, 1,:], swf[ii,1,:], color='darkgray')
            ax.plot(swk[ii, 2,:], swf[ii,2,:], color='darkgray')

        ax.axvline(0, linestyle='dashed', color='lightgray')

    # ax.set_xlabel('Wavenumber')
    # ax.set_ylabel('Frequency')
    ax.set_title(tlt)
    # ax.set_yticks([0.1, 0.2, 0.3, 0.4])
    # ax.set_xticks([-15, -10, -5, 0, 5, 10, 15])
    ax.set_xlim(-15, 15)
    ax.set_ylim(fb)
    # Customize tick labels
    ax.tick_params(axis='both', labelsize=ftsize, which='major', length=10, width=1.5)
    # ax.grid(visible=True, linestyle='dashed', color='lightgray')

    cax = fig.add_subplot(gs[0, 1])

    cbar = plt.colorbar(c, cax=cax)

    # if logflg:
    #     cbar = plt.colorbar(c, cax=cax, orientation='horizontal')
    # else:
    #     cbar = plt.colorbar(c, cax=cax, orientation='horizontal', ticks=np.linspace(0, 4, 9), boundaries=np.linspace(0, 4, 100))

    cbar.ax.tick_params(labelsize=ftsize)

    # # Create a twin axis for the second y-axis
    # ax2 = ax.twinx()
    # # ax2.set_ylabel('Period (days)')

    # # Modify tick labels for the second y-axis
    # ax2.set_ylim(ax.get_ylim())  # Match the y-axis limits with the first y-axis
    # ax2.set_yticks(ax.get_yticks())
    # ax2.set_yticklabels([f'{1/freq:.1f}' if freq != 0 else '0' for freq in ax.get_yticks()])
    # # Customize tick labels
    # ax2.tick_params(axis='both', labelsize=22)
    # plt.tick_params(axis='both', which='major', length=10, width=2)  # Length of major ticks
    # plt.tick_params(axis='both', which='minor', length=5)   # Length of minor ticks
    
    # Adjust the spacing between subplots and colorbars
    plt.subplots_adjust(wspace=0.1)

    # Save the figure if savflg is True
    if savflg:
        plt.savefig(filename, bbox_inches='tight')
    # Show the plot
    plt.show()
import numpy as np
import matplotlib.pyplot as plt

def wk_plot_sym_hid_one(ax, sym, tlt='', logflg=True, pltDispCurve=True, fb=[0,0.48], center0=False, setcolor=False, vmax=None, vmin=None, cmapflg='Blues'):
    wavenumber = sym['wavenumber']
    frequency = sym['frequency']

    if logflg:
        if center0:
            vmax = np.max(np.abs(np.log10(sym.T)))
            c = ax.contourf(wavenumber, frequency, np.log10(sym.T), cmap='RdBu_r', levels=15, vmin=-vmax, vmax=vmax)
        else:
            if setcolor:
                c = ax.contourf(wavenumber, frequency, np.log10(sym.T), cmap=cmapflg, vmin=vmin, vmax=vmax, levels=8)
            else:
                c = ax.contourf(wavenumber, frequency, np.log10(sym.T), cmap=cmapflg, levels=8)
    else:
        if setcolor:
            c = ax.contourf(wavenumber, frequency, sym.T, vmin=vmin, vmax=vmax, levels=8, cmap=cmapflg)
        else:
            c = ax.contourf(wavenumber, frequency, sym.T, levels=8, cmap=cmapflg)

    if pltDispCurve:
        wavfreq, wavwn = genDispersionCurves()
        swf = np.where(wavfreq == 1e20, np.nan, wavfreq)
        swk = np.where(wavwn == 1e20, np.nan, wavwn)

        for ii in range(3,6):
            ax.plot(swk[ii, 0,:], swf[ii,0,:], color='darkgray')
            ax.plot(swk[ii, 1,:], swf[ii,1,:], color='darkgray')
            ax.plot(swk[ii, 2,:], swf[ii,2,:], color='darkgray')

        ax.axvline(0, linestyle='dashed', color='lightgray')

    ax.set_title(tlt)
    ax.set_xlim(-15, 15)
    ax.set_ylim(fb)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4])
    ax.set_xticks([-15, -10, -5, 0, 5, 10, 15])
    ax.tick_params(axis='both', which='major', length=10, width=1.5)
    # Add colorbar for each subplot
    cbar = plt.colorbar(c, ax=ax)
    cbar.ax.tick_params(labelsize=15)
    return c

def wk_plot_asym(asym, tlt='', logflg=True, savflg=False):
    wavenumber = asym['wavenumber']
    frequency = asym['frequency']

    plt.rcParams.update({'font.size': 18})

    # Create a figure and subplots
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.05], width_ratios=[1, 1])

    # Plot smooth_sym for each frequency
    ax = fig.add_subplot(gs[0, 0])
    if logflg:
        c = ax.contourf(wavenumber, frequency, np.log10(asym.T), cmap='Reds', levels=10)
    else:
        c = ax.contourf(wavenumber, frequency, asym.T, cmap='Reds', levels=10)

    ax.set_xlabel('Wavenumber')
    ax.set_ylabel('Frequency')
    ax.set_title(tlt)
    ax.set_xlim(-15, 15)
    ax.set_yticks([0.05, 0.1, 0.2, 0.3, 0.4])
    # Customize tick labels
    ax.tick_params(axis='both', labelsize=12)

    cax = fig.add_subplot(gs[1, 0])
    cbar = plt.colorbar(c, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=12)

    # Create a twin axis for the second y-axis
    ax2 = ax.twinx()
    ax2.set_ylabel('Period (days)')

    # Modify tick labels for the second y-axis
    ax2.set_ylim(ax.get_ylim())  # Match the y-axis limits with the first y-axis
    ax2.set_yticks(ax.get_yticks())
    ax2.set_yticklabels([f'{1/freq:.1f}' if freq != 0 else '0' for freq in ax.get_yticks()])
    # Customize tick labels
    ax2.tick_params(axis='both', labelsize=12)

    # Adjust the spacing between subplots and colorbars
    plt.subplots_adjust(wspace=0.4, hspace=0.3)

    # Save the figure if savflg is True
    if savflg:
        plt.savefig('asym_power.png', dpi=300, bbox_inches='tight')
    # Show the plot
    plt.show()

def wk_plot_symsig(sym, symsig, tlt='', savflg=False):

    # Assuming smooth_sym and smooth_asym are DataArrays with dimensions (wavenumber, frequency)
    wavenumber = sym['wavenumber']
    frequency = sym['frequency']

    plt.rcParams.update({'font.size': 18})

    # Create a figure and subplots
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.05], width_ratios=[1, 1])

    # Plot smooth_sym for each frequency
    ax = fig.add_subplot(gs[0, 0])
    c0 = ax.contour(wavenumber, frequency, symsig.T, colors='black', levels=15)
    # ax.clabel(c0, inline=True, fontsize=10)
    c = ax.contourf(wavenumber, frequency, np.log10(sym.T), cmap='coolwarm', levels=10, vmin=-1, vmax=1)

    ax.set_xlabel('Wavenumber')
    ax.set_ylabel('Frequency')
    ax.set_title(tlt)
    ax.set_xlim(-15, 15)
    ax.set_yticks([0.05, 0.1, 0.2, 0.3, 0.4])
    # Customize tick labels
    ax.tick_params(axis='both', labelsize=12)

    cax = fig.add_subplot(gs[1, 0])
    cbar = plt.colorbar(c, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=12)

    # Create a twin axis for the second y-axis
    ax2 = ax.twinx()
    ax2.set_ylabel('Period (days)')

    # Modify tick labels for the second y-axis
    ax2.set_ylim(ax.get_ylim())  # Match the y-axis limits with the first y-axis
    ax2.set_yticks(ax.get_yticks())
    ax2.set_yticklabels([f'{1/freq:.1f}' if freq != 0 else '0' for freq in ax.get_yticks()])
    # Customize tick labels
    ax2.tick_params(axis='both', labelsize=12)

    # Adjust the spacing between subplots and colorbars
    plt.subplots_adjust(wspace=0.4, hspace=0.3)

    # Save the figure if savflg is True
    if savflg:
        plt.savefig('normpower_symsig.png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

def wk_plot_asymsig(asym, asymsig, tlt='', savflg=False):

    # Assuming smooth_sym and smooth_asym are DataArrays with dimensions (wavenumber, frequency)
    wavenumber = asym['wavenumber']
    frequency = asym['frequency']

    plt.rcParams.update({'font.size': 18})

    # Create a figure and subplots
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.05], width_ratios=[1, 1])

    # Plot smooth_sym for each frequency
    ax = fig.add_subplot(gs[0, 0])
    c0 = ax.contour(wavenumber, frequency, asym.T, colors='black')
    ax.clabel(c0, inline=True, fontsize=10)
    c = ax.contourf(wavenumber, frequency, asymsig.T, cmap='Reds', levels=10)

    ax.set_xlabel('Wavenumber')
    ax.set_ylabel('Frequency')
    ax.set_title(tlt)
    ax.set_xlim(-15, 15)
    ax.set_yticks([0.05, 0.1, 0.2, 0.3, 0.4])
    # Customize tick labels
    ax.tick_params(axis='both', labelsize=12)

    cax = fig.add_subplot(gs[1, 0])
    cbar = plt.colorbar(c, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=12)

    # Create a twin axis for the second y-axis
    ax2 = ax.twinx()
    ax2.set_ylabel('Period (days)')

    # Modify tick labels for the second y-axis
    ax2.set_ylim(ax.get_ylim())  # Match the y-axis limits with the first y-axis
    ax2.set_yticks(ax.get_yticks())
    ax2.set_yticklabels([f'{1/freq:.1f}' if freq != 0 else '0' for freq in ax.get_yticks()])
    # Customize tick labels
    ax2.tick_params(axis='both', labelsize=12)

    # Adjust the spacing between subplots and colorbars
    plt.subplots_adjust(wspace=0.4, hspace=0.3)

    # Save the figure if savflg is True
    if savflg:
        plt.savefig('normpower_asymsig.png', dpi=300, bbox_inches='tight')
    # Show the plot
    plt.show()

def wk_plot_bag(bag, logflg=True, savflg=False):
    # Assuming smooth_sym and smooth_asym are DataArrays with dimensions (wavenumber, frequency)
    wavenumber = bag['wavenumber']
    frequency = bag['frequency']

    plt.rcParams.update({'font.size': 18})

    # Create a figure and subplots
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.05], width_ratios=[1, 1])

    # Plot smooth_sym for each frequency
    ax = fig.add_subplot(gs[0, 0])
    if logflg:
        c = ax.contourf(wavenumber, frequency, np.log10(bag.T), cmap='Reds', levels=10)
    else:
        c = ax.contourf(wavenumber, frequency, bag.T, cmap='Reds', levels=10)

    ax.set_xlabel('Wavenumber')
    ax.set_ylabel('Frequency')
    ax.set_title('log(Background)')
    ax.set_xlim(-15, 15)
    ax.set_yticks([0.05, 0.1, 0.2, 0.3, 0.4])
    # Customize tick labels
    ax.tick_params(axis='both', labelsize=12)

    cax = fig.add_subplot(gs[1, 0])
    cbar = plt.colorbar(c, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=12)

    # Create a twin axis for the second y-axis
    ax2 = ax.twinx()
    ax2.set_ylabel('Period (days)')

    # Modify tick labels for the second y-axis
    ax2.set_ylim(ax.get_ylim())  # Match the y-axis limits with the first y-axis
    ax2.set_yticks(ax.get_yticks())
    ax2.set_yticklabels([f'{1/freq:.1f}' if freq != 0 else '0' for freq in ax.get_yticks()])
    # Customize tick labels
    ax2.tick_params(axis='both', labelsize=12)

    # Adjust the spacing between subplots and colorbars
    plt.subplots_adjust(hspace=0.3)

    # Save the figure if savflg is True
    if savflg:
        plt.savefig('background.png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

# get MJO-related signals by filtering the Wheeler-Kiladis spectra
def detrend_func(x):
    return detrend(x, axis=0)



def filter_olr(data, spd=1, lat_lim=15, remove_low=True, kmin=1, kmax=5, flow=1/100, fhig=1/20):
    """
    kmin: minimum wavenumber to keep; it should be the left boundary of the wavenumber range
    kmax: maximum wavenumber to keep; it should be the right boundary of the wavenumber range
    Perform space-time spectral decomposition and return raw power spectrum following Wheeler-Kiladis approach.

    data: an xarray DataArray to be analyzed; needs to have (time, lat, lon) dimensions.
    spd: sampling rate, in "samples per day" (e.g. daily=1, 6-houry=4)

    Method
    ------------------
        0. Remove the first three harmonics of annual cycle to prevent aliasing. data[time, lat, lon]
        1. Subsample in latitude if latitude_bounds is specified.
        2. Construct symmetric/antisymmetric array .

        4. Detrend the segments (remove linear trend).
        5. Apply taper in time dimension of windows (aka segments).
        6. Fourier transform
        7. filter the fft coefficients
        8. Inverse Fourier transform
        9. average over all latitudes to create a hovalmoller diagram  
    """

    segsize = spd * data.sizes['time']  # how many time steps included.
    # print('segsize: ', segsize)
    # noverlap = spd * noverlap  

    # select the interested section in data
    data1 = data.sel(lat=slice(lat_lim, -lat_lim)).load()

    # 0. remove low-frequency signals
    if remove_low:
        data2 = rmv_lowfreq(data1)
    else:
        data2 = data1

    # 2. [time, lat(pos+neg), lon]
    data_sym_asym = decompose2SymAsym(data2)

    # print('shape of data_sym_asym: ', data_sym_asym.shape)
    # 4. 

    def detrend_timeseries(ts):
        time_nums = np.arange(ts.size)
        coeffs = np.polyfit(time_nums, ts, 1)
        trend = np.polyval(coeffs, time_nums)
        return ts - trend

    # Use apply_ufunc to apply the detrending
    detrended_data = xr.apply_ufunc(
        detrend_timeseries,
        data_sym_asym,  # Your Dask-backed DataArray
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[data_sym_asym.dtype]
    )

    # print('detrended_data: ', detrended_data.shape)
    x_detrend_sym_asym = detrended_data.compute()
    x_detrend_sym_asym = x_detrend_sym_asym.transpose('time', 'lat', 'lon')
    # print('shape of detrended data: ', x_detrend_sym_asym.shape)

    # 5. add taper
    # using hanning window w[n] = 0.5 * (1 - cos(2*pi*n/(M-1))) to create split cosine bell taper
    taper = split_hann_taper_pnt(seg_size=segsize, pnt=20)
    taper = xr.DataArray(taper, dims=['time'], coords={'time': detrended_data['time']})
    
    # print('shape of taper: ', np.shape(taper))

    x_detrend_tap = x_detrend_sym_asym * taper  # time, lat, lon
    # x_detrend_asymtap = x_detrend_asym * taper

    # 6. fourier transform within 2 steps 
    # perform fft in longitude
    lon_dim = x_detrend_tap.dims.index('lon')
    fft_lon = np.fft.fft(x_detrend_tap, axis=lon_dim)  # time, lat, wavenumber
    coef_flt = np.zeros(fft_lon.shape, dtype=complex)
    
    # 7. filter the fft coefficients
    wavenum = np.fft.fftfreq(len(x_detrend_tap['lon']), d=1/len(x_detrend_tap['lon']))
    kmin1 = np.argmin(np.abs(wavenum - kmin))
    kmax1 = np.argmin(np.abs(wavenum - kmax))
    print('kmin: ', kmin1, 'kmax: ', kmax1)

    coef_flt[:, :, kmin1:kmax1+1] = fft_lon[:, :, kmin1:kmax1+1]  # time, lat, wavenumber

    kmin2 = np.argmin(np.abs(wavenum + kmax))
    kmax2 = np.argmin(np.abs(wavenum + kmin))
    print('kmin: ', kmin2, 'kmax: ', kmax2)

    coef_flt[:, :, kmin2:kmax2+1] = fft_lon[:, :, kmin2:kmax2+1]  # time, lat, wavenumber; symmetric

    # fft_lon[:, :, :kmin] = 0.0
    # fft_lon[:, :, kmax+1:-kmax] = 0.0
    # if kmin > 1: 
    #     fft_lon[:, :, -kmin+1:] = 0.0

    seg_dim = x_detrend_tap.dims.index('time')
    fft_lonseg = np.fft.fft(coef_flt, axis=seg_dim) 

    freq = np.fft.fftfreq(len(x_detrend_tap['time']), d=1/spd)
    tlow = np.argmin(np.abs(freq - flow))
    thig = np.argmin(np.abs(freq - fhig))  # flow and fhig are always positive 

    fft_lonseg[:tlow, :, :] = 0.0
    fft_lonseg[:-thig, :, kmin1:kmax1+1] = 0.0
    fft_lonseg[thig+1:, :, kmin2:kmax2+1] = 0.0

    if tlow > 1:
        fft_lonseg[-tlow+1:, :, :] = 0.0

    # 8. Inverse Fourier transform
    x_lon = np.fft.ifft(fft_lonseg, axis=seg_dim)
    x_filtered = np.real(np.fft.ifft(x_lon, axis=lon_dim))

    x_filtered = xr.DataArray(
        x_filtered, 
        dims=['time','lat', 'lon'], 
        coords={'time': x_detrend_tap['time'], 'lat': x_detrend_tap['lat'], 'lon': x_detrend_tap['lon']}
    )

    return x_filtered


def get_MJO_signal(u, d=1, kmin=1, kmax=3, flow=1/100.0, fhig=1/20.0, detrendflg=True):
    '''
    0. input u=u[time, lon]
    1. optional: detrend the data in time. Defult is True.
    2. apply taper in time
    3. Fourier transform in time and space.
    4. remove coefficients outside k=1-3, T=20-100 day
    5. reconstruct u 
    '''

    # detrend
    if detrendflg:
        u_detrended = xr.apply_ufunc(
            detrend_func,
            u,
            # input_core_dims=[['time']],
            # output_core_dims=[['time']],
            dask="parallelized",
            output_dtypes=[u.dtype],
        )
    else:
        u_detrended = u.copy()

    # taper
    # using hanning window w[n] = 0.5 * (1 - cos(2*pi*n/(M-1))) to create split cosine bell taper
    taper = split_hann_taper_pnt(seg_size=len(u['time']))

    u_detrend_tap = u_detrended * xr.DataArray(taper, dims=['time'])

    # Fourier transform
    lon_index = u.dims.index('lon')
    u_lon = np.fft.fft(u_detrend_tap.values, axis=lon_index)  # [time, k]

    u_lon[:,:kmin] = 0.0
    u_lon[:,kmax+1:-kmax] = 0.0

    time_index = u.dims.index('time')
    u_lon_time = np.fft.fft(u_lon, axis=time_index)

    freq = np.fft.fftfreq(len(u['time']), d)
    tlow = np.argmin(np.abs(freq - flow))
    thig = np.argmin(np.abs(freq - fhig))

    if tlow == 1:
        u_lon_time[:tlow,:] = 0.0
        u_lon_time[thig+1:-thig,:] = 0.0
        # u_lon_time[-tlow+1:,:] = 0.0

        u_lon_time[tlow:thig+1, kmin:kmax+1] = 0.0
        u_lon_time[-thig:, -kmax:] = 0.0

    else:
        u_lon_time[:tlow,:] = 0.0
        u_lon_time[thig+1:-thig,:] = 0.0
        u_lon_time[-tlow+1:,:] = 0.0

        u_lon_time[tlow:thig+1, kmin:kmax+1] = 0.0
        u_lon_time[-thig:-tlow+1, -kmax:] = 0.0

    # reconstruct u
    u_retime = np.fft.ifft(u_lon_time, axis=time_index)
    u_re_values = np.fft.ifft(u_retime, axis=lon_index)

    u_re = xr.DataArray(
        data=u_re_values.real,
        dims=u.dims,
        coords=u.coords,
    )

    return u_re

# def get_MJO_signal_3D(u, d=1, kmin=1, kmax=3, flow=1/100.0, fhig=1/20.0, detrendflg=True):
#     '''
#     0. input u=u[time, lat, lon]
#     1. detrend the data.
#     2. apply taper in time
#     3. Fourier transform
#     4. remove coefficients outside k=1-5, T=20-100 day
#     5. reconstruct u 
#     '''

#     # detrend
#     if detrendflg:
#         u_detrended = xr.apply_ufunc(
#             detrend_func,
#             u,
#             # input_core_dims=[['time']],
#             # output_core_dims=[['time']],
#             dask="parallelized",
#             output_dtypes=[u.dtype],
#         )
#     else:
#         u_detrended = u.copy()

#     # taper
#     # using hanning window w[n] = 0.5 * (1 - cos(2*pi*n/(M-1))) to create split cosine bell taper
#     taper = split_hann_taper(seg_size=len(u['time']), fraction=0.5)

#     u_detrend_tap = u_detrended * xr.DataArray(taper, dims=["time"])

#     # Fourier transform
#     u_lon = np.fft.fft(u_detrend_tap.values, axis=-1)
#     u_lon[:,:,:kmin] = 0.0
#     u_lon[:,:,kmax+1:-kmax] = 0.0

#     u_lon_time = np.fft.fft(u_lon, axis=0)

#     freq = np.fft.fftfreq(len(u['time']), d)
#     tlow = np.argmin(np.abs(freq - flow))
#     thig = np.argmin(np.abs(freq - fhig))

#     u_lon_time[:tlow,:,:] = 0.0
#     u_lon_time[thig+1:-thig,:,:] = 0.0
#     u_lon_time[-tlow+1:,:,:] = 0.0

#     u_lon_time[tlow:thig+1,:, kmin:kmax+1] = 0.0
#     u_lon_time[-thig:-tlow+1,:, -kmax:] = 0.0

#     # reconstruct u
#     u_retime = np.fft.ifft(u_lon_time, axis=0)
#     u_re_values = np.fft.ifft(u_retime, axis=-1)

#     u_re = xr.DataArray(
#         data=u_re_values.real,
#         dims=u.dims,
#         coords=u.coords,
#     )

#     return u_re

# def get_MJO_signal_4D(u, d=1, kmin=1, kmax=3, flow=1/100.0, fhig=1/20.0, detrendflg=True):
#     '''
#     0. input u=u[time, lev, lat, lon] 
#     1. detrend the data.
#     2. apply taper in time
#     3. Fourier transform
#     4. remove coefficients outside k=1-5, T=20-100 day
#     5. reconstruct u 
#     '''

#     # detrend
#     if detrendflg:
#         u_detrended = xr.apply_ufunc(
#             detrend_func,
#             u,
#             # input_core_dims=[['time']],
#             # output_core_dims=[['time']],
#             dask="parallelized",
#             output_dtypes=[u.dtype],
#         )
#     else:
#         u_detrended = u.copy()

#     # taper
#     # using hanning window w[n] = 0.5 * (1 - cos(2*pi*n/(M-1))) to create split cosine bell taper
#     taper = split_hann_taper(seg_size=len(u['time']), fraction=0.5)

#     u_detrend_tap = u_detrended * xr.DataArray(taper, dims=["time"])

#     # Fourier transform
#     u_lon = np.fft.fft(u_detrend_tap.values, axis=-1)
#     u_lon[:,:,:,:kmin] = 0.0
#     u_lon[:,:,:,kmax+1:-kmax] = 0.0

#     u_lon_time = np.fft.fft(u_lon, axis=0)

#     freq = np.fft.fftfreq(len(u['time']), d)
#     tlow = np.argmin(np.abs(freq - flow))
#     thig = np.argmin(np.abs(freq - fhig))

#     u_lon_time[:tlow,:,:,:] = 0.0
#     u_lon_time[thig+1:-thig,:,:,:] = 0.0
#     u_lon_time[-tlow+1:,:,:,:] = 0.0

#     u_lon_time[tlow:thig+1,:,:, kmin:kmax+1] = 0.0
#     u_lon_time[-thig:-tlow+1,:,:, -kmax:] = 0.0

#     # reconstruct u
#     u_retime = np.fft.ifft(u_lon_time, axis=0)
#     u_re_values = np.fft.ifft(u_retime, axis=-1)

#     u_re = xr.DataArray(
#         data=u_re_values.real,
#         dims=u.dims,
#         coords=u.coords,
#     )

#     return u_re

def get_local_MSE(ds, lat_lim=50, latmean=False, iceflg=False):
    # ds is the 3D model output; daily data
    # iceflg indicates if we include the fusion of ice

    # constants
    cp = 1004 
    g = 9.8
    lv = 2.5104e6
    lf = 0.3336e6
    if latmean:
        T = ds['T'].sel(lat=slice(-lat_lim, lat_lim)).mean('lat').load()
        z = ds['Z3'].sel(lat=slice(-lat_lim, lat_lim)).mean('lat').load()
        qv = ds['Q'].sel(lat=slice(-lat_lim, lat_lim)).mean('lat').load()

        mse = cp * T + g * z + lv * qv
    else:
        T = ds['T'].sel(lat=slice(-lat_lim, lat_lim)).load()
        z = ds['Z3'].sel(lat=slice(-lat_lim, lat_lim)).load()
        qv = ds['Q'].sel(lat=slice(-lat_lim, lat_lim)).load()

        mse = cp * T + g * z + lv * qv
    return mse

def get_local_MSE_sep(ds, lat_lim=50, latmean=False, iceflg=False):
    # ds is the 3D model output; daily data
    # iceflg indicates if we include the fusion of ice

    # constants
    cp = 1004 
    g = 9.8
    lv = 2.5104e6
    lf = 0.3336e6

    if latmean:
        T = ds['T'].sel(lat=slice(-lat_lim, lat_lim)).mean('lat').load()
        z = ds['Z3'].sel(lat=slice(-lat_lim, lat_lim)).mean('lat').load()
        qv = ds['Q'].sel(lat=slice(-lat_lim, lat_lim)).mean('lat').load()

        dse = cp * T + g * z 
        qlv = lv * qv
    else:
        T = ds['T'].sel(lat=slice(-lat_lim, lat_lim)).load()
        z = ds['Z3'].sel(lat=slice(-lat_lim, lat_lim)).load()
        qv = ds['Q'].sel(lat=slice(-lat_lim, lat_lim)).load()

        dse = cp * T + g * z 
        qlv = lv * qv
    return dse, qlv

def get_integrated_MSE(ds, lat_lim=50, plim=None, latmean=False, iceflg=False):
    
    local_mse = get_local_MSE(ds=ds, lat_lim=lat_lim, latmean=latmean, iceflg=iceflg)
    # [time, lev, lat, lon] or [time, lev, lon]

    # integration over pressure
    # int_mse = sum(local_mse * dp) / g 
    p = local_mse['lev'].values
    dp = np.zeros(len(p))

    dp[1:] = p[1:] - p[0:-1]
    dp[0] = dp[1]  # hPa
    g = 9.8

    # find the tropopause by T
    if plim is None:
        T = ds['T'].sel(lat=slice(-10, 10)).mean(dim=["time","lat","lon"]).values
        tp_index = np.argmin(T)
        if latmean:
            int_mse = (local_mse[:,tp_index:,:] * np.reshape(dp[tp_index:], (1, len(dp[tp_index:]), 1)) / g * 100).sum(dim='lev').squeeze()
        else:
            int_mse = (local_mse[:,tp_index:,:,:] * np.reshape(dp[tp_index:], (1, len(dp[tp_index:]),1, 1)) / g * 100).sum(dim='lev').squeeze()
    
    else:
        tp_index = np.argmin(np.abs(p - plim))
        if latmean:
            int_mse = (local_mse[:,tp_index:,:] * np.reshape(dp[tp_index:], (1, len(dp[tp_index:]), 1)) / g * 100).sum(dim='lev').squeeze()
        else:
            int_mse = (local_mse[:,tp_index:,:,:] * np.reshape(dp[tp_index:], (1, len(dp[tp_index:]), 1, 1)) / g * 100).sum(dim='lev').squeeze()
    
    return int_mse

def get_local_MSE_tendency(ds, lat_lim=50, latmean=False, iceflg=False):
    local_tend = ds['DDSE_TOT'].sel(lat=slice(-lat_lim, lat_lim)).load() + ds['DQLV_TOT'].sel(lat=slice(-lat_lim, lat_lim)).load()
    
    if latmean:
        return local_tend.mean('lat')
    else:
        return local_tend

def get_local_MSE_tendency_sep(ds, lat_lim=50, latmean=False, iceflg=False):
    local_tend_dse = ds['DDSE_TOT'].sel(lat=slice(-lat_lim, lat_lim)).load()
    local_tend_qlv = ds['DQLV_TOT'].sel(lat=slice(-lat_lim, lat_lim)).load()
    
    if latmean:
        return local_tend_dse.mean('lat'), local_tend_qlv.mean('lat')
    else:
        return local_tend_dse, local_tend_qlv
    
def get_local_MSE_source(ds, varn, lat_lim=50, latmean=False, iceflg=False):
    local_src = ds[varn].sel(lat=slice(-lat_lim, lat_lim)).load() 
    if latmean:
        return local_src.mean('lat')
    else:
        return local_src

def get_local_MSE_budget(ds, lat_lim=50, plim=100, latmean=False, iceflg=False):
    mse_budget = {}

    # get local MSE
    mse_sel = get_local_MSE(ds, lat_lim=lat_lim, latmean=latmean).load().copy()  # [time, lev, lat, lon]
    raw_mse = mse_sel.sel(lev=slice(plim,None)) 
    mse_budget['mse'] = raw_mse

    dtmse_sel = get_local_MSE_tendency(ds, lat_lim=lat_lim, latmean=latmean).load().copy() # [lev, lat, lon]
    raw_dtmse = dtmse_sel.sel(lev=slice(plim,None))
    mse_budget['tendency'] = raw_dtmse * 86400

    # get local MSE source: CRM
    dtmse_sel = (get_local_MSE_source(ds,'DDSE_CRM',lat_lim=lat_lim, latmean=latmean).load().copy()
            + get_local_MSE_source(ds,'DQLV_CRM',lat_lim=lat_lim, latmean=latmean).load().copy())
    raw_dtmse = dtmse_sel.sel(lev=slice(plim,None))
    mse_budget['crm'] = raw_dtmse * 86400

    # get local MSE source: CRM_ALT
    dtmse_sel = (get_local_MSE_source(ds,'DDSE_CRM_ALT',lat_lim=lat_lim, latmean=latmean).load().copy()
            + get_local_MSE_source(ds,'DQLV_CRM_ALT',lat_lim=lat_lim, latmean=latmean).load().copy()
            - get_local_MSE_source(ds,'DDSE_QRS',lat_lim=lat_lim, latmean=latmean).load().copy()
            - get_local_MSE_source(ds,'DDSE_QRL',lat_lim=lat_lim, latmean=latmean).load().copy())
    raw_dtmse = dtmse_sel.sel(lev=slice(plim,None))
    mse_budget['crmalt'] = raw_dtmse * 86400

    # get local MSE source: PBL
    dtmse_sel = (get_local_MSE_source(ds,'DDSE_PBL',lat_lim=lat_lim, latmean=latmean).load().copy()
            + get_local_MSE_source(ds,'DQLV_PBL',lat_lim=lat_lim, latmean=latmean).load().copy())
    raw_dtmse = dtmse_sel.sel(lev=slice(plim,None))
    mse_budget['pbl'] = raw_dtmse * 86400


    dtmse_sel = (get_local_MSE_source(ds,'DDSE_QRS',lat_lim=lat_lim, latmean=latmean).load().copy() 
                 + get_local_MSE_source(ds,'DDSE_QRL',lat_lim=lat_lim, latmean=latmean).load().copy())
    raw_dtmse = dtmse_sel.sel(lev=slice(plim,None))
    mse_budget['qr'] = raw_dtmse * 86400

    dtmse_sel = (get_local_MSE_source(ds,'DDSE_DYN',lat_lim=lat_lim, latmean=latmean).load().copy() 
                 + get_local_MSE_source(ds,'DQLV_DYN',lat_lim=lat_lim, latmean=latmean).load().copy())
    raw_dtmse = dtmse_sel.sel(lev=slice(plim,None))
    mse_budget['dyn'] = raw_dtmse * 86400

    return mse_budget

def get_local_MSE_budget_sep(ds, lat_lim=50, plim=100, latmean=False, iceflg=False):
    mse_budget = {}

    # get local MSE
    dse_sel, qlv_sel = get_local_MSE_sep(ds, lat_lim=lat_lim, latmean=latmean)  # [time, lev, lat, lon]
    raw_dse = dse_sel.sel(lev=slice(plim,None)) 
    raw_qlv = qlv_sel.sel(lev=slice(plim,None))
    mse_budget['dse'] = raw_dse
    mse_budget['qlv'] = raw_qlv
    print('get local MSE')

    dtdse_sel, dtqlv_sel = get_local_MSE_tendency_sep(ds, lat_lim=lat_lim, latmean=latmean) # [lev, lat, lon]
    raw_dtdse = dtdse_sel.sel(lev=slice(plim,None))
    raw_dtqlv = dtqlv_sel.sel(lev=slice(plim,None))
    mse_budget['dtdse'] = raw_dtdse * 86400  # J/kg/day
    mse_budget['dtqlv'] = raw_dtqlv * 86400
    print('get local MSE tendency')

    # get local MSE source: CRM_ALT
    dtdse_sel = (get_local_MSE_source(ds,'DDSE_CRM_ALT',lat_lim=lat_lim, latmean=latmean).load()
            - get_local_MSE_source(ds,'DDSE_QRS',lat_lim=lat_lim, latmean=latmean).load()
            - get_local_MSE_source(ds,'DDSE_QRL',lat_lim=lat_lim, latmean=latmean).load())
    dtqlv_sel = get_local_MSE_source(ds,'DQLV_CRM_ALT',lat_lim=lat_lim, latmean=latmean).load()
    raw_dtdse = dtdse_sel.sel(lev=slice(plim,None))
    raw_dtqlv = dtqlv_sel.sel(lev=slice(plim,None))
    mse_budget['crm_dse'] = raw_dtdse * 86400
    mse_budget['crm_qlv'] = raw_dtqlv * 86400
    print('get local MSE source: CRM_ALT')

    # get local MSE source: PBL
    dtdse_sel = get_local_MSE_source(ds,'DDSE_PBL',lat_lim=lat_lim, latmean=latmean).load()
    dtqlv_sel = get_local_MSE_source(ds,'DQLV_PBL',lat_lim=lat_lim, latmean=latmean).load()
    raw_dtdse = dtdse_sel.sel(lev=slice(plim,None))
    raw_dtqlv = dtqlv_sel.sel(lev=slice(plim,None))
    mse_budget['pbl_dse'] = raw_dtdse * 86400
    mse_budget['pbl_qlv'] = raw_dtqlv * 86400
    print('get local MSE source: PBL')

    # get local MSE source: QR
    dtdse_sel = (get_local_MSE_source(ds,'DDSE_QRS',lat_lim=lat_lim, latmean=latmean).load()
                 + get_local_MSE_source(ds,'DDSE_QRL',lat_lim=lat_lim, latmean=latmean).load())
    raw_dtdse = dtdse_sel.sel(lev=slice(plim,None))
    mse_budget['qr'] = raw_dtdse * 86400
    print('get local MSE source: QR')

    # get local MSE source: DYN
    dtdse_sel = get_local_MSE_source(ds,'DDSE_DYN',lat_lim=lat_lim, latmean=latmean).load()
    dtqlv_sel = get_local_MSE_source(ds,'DQLV_DYN',lat_lim=lat_lim, latmean=latmean).load()
    raw_dtdse = dtdse_sel.sel(lev=slice(plim,None))
    raw_dtqlv = dtqlv_sel.sel(lev=slice(plim,None))
    mse_budget['dyn_dse'] = raw_dtdse * 86400
    mse_budget['dyn_qlv'] = raw_dtqlv * 86400
    print('get local MSE source: DYN')
    return mse_budget

def get_local_MSE_budget_old(ds, lat_lim=5, plim=100, latmean=True, iceflg=False):
    mse_budget = {}

    # get local MSE
    mse_sel = get_local_MSE(ds, lat_lim=lat_lim).load().copy()  # [time, lev, lat, lon]
    if latmean:
        raw_mse = mse_sel.sel(lev=slice(plim,None)).mean('lat')  # [time, lev, lon]   
    else:
        raw_mse = mse_sel.sel(lev=slice(plim,None)) 
    mse_budget['mse'] = raw_mse

    dtmse_sel = get_local_MSE_tendency(ds, lat_lim=lat_lim).load().copy() # [lev, lat, lon]
    if latmean:
        raw_dtmse = dtmse_sel.sel(lev=slice(plim,None)).mean(['lat'])  # [lev, lon]
    else:
        raw_dtmse = dtmse_sel.sel(lev=slice(plim,None))
    mse_budget['tendency'] = raw_dtmse * 86400

    # get local MSE source: CRM
    dtmse_sel = (get_local_MSE_source(ds,'DDSE_CRM',lat_lim=lat_lim).load().copy()
            + get_local_MSE_source(ds,'DQLV_CRM',lat_lim=lat_lim).load().copy()
            + get_local_MSE_source(ds,'DDSE_PBL',lat_lim=lat_lim).load().copy()
            + get_local_MSE_source(ds,'DQLV_PBL',lat_lim=lat_lim).load().copy()
            - get_local_MSE_source(ds,'DDSE_QRS',lat_lim=lat_lim).load().copy()
            - get_local_MSE_source(ds,'DDSE_QRL',lat_lim=lat_lim).load().copy())
    if latmean:
        raw_dtmse = dtmse_sel.sel(lev=slice(plim,None)).mean('lat')  # [lev, lon]
    else:
        raw_dtmse = dtmse_sel.sel(lev=slice(plim,None))
    mse_budget['crm'] = raw_dtmse * 86400

    # get local MSE source: CRM_ALT
    dtmse_sel = (get_local_MSE_source(ds,'DDSE_CRM_ALT',lat_lim=lat_lim).load().copy()
            + get_local_MSE_source(ds,'DQLV_CRM_ALT',lat_lim=lat_lim).load().copy()
            - get_local_MSE_source(ds,'DDSE_QRS',lat_lim=lat_lim).load().copy()
            - get_local_MSE_source(ds,'DDSE_QRL',lat_lim=lat_lim).load().copy())
    if latmean:
        raw_dtmse = dtmse_sel.sel(lev=slice(plim,None)).mean('lat')  # [lev, lon]
    else:
        raw_dtmse = dtmse_sel.sel(lev=slice(plim,None))
    mse_budget['crmalt'] = raw_dtmse * 86400

    # get local MSE source: CRM + PBL
    dtmse_sel = (get_local_MSE_source(ds,'DDSE_PBL',lat_lim=lat_lim).load().copy()
            + get_local_MSE_source(ds,'DQLV_PBL',lat_lim=lat_lim).load().copy())
    if latmean:
        raw_dtmse = dtmse_sel.sel(lev=slice(plim,None)).mean('lat')  # [lev, lon]
    else:
        raw_dtmse = dtmse_sel.sel(lev=slice(plim,None))
    mse_budget['pbl'] = raw_dtmse * 86400


    dtmse_sel = (get_local_MSE_source(ds,'DDSE_QRS',lat_lim=lat_lim).load().copy() 
                 + get_local_MSE_source(ds,'DDSE_QRL',lat_lim=lat_lim).load())
    if latmean:
        raw_dtmse = dtmse_sel.sel(lev=slice(plim,None)).mean('lat')  # [lev, lon]
    else:
        raw_dtmse = dtmse_sel.sel(lev=slice(plim,None))
    mse_budget['qr'] = raw_dtmse * 86400

    dtmse_sel = (get_local_MSE_source(ds,'DDSE_DYN',lat_lim=lat_lim).load().copy() 
                 + get_local_MSE_source(ds,'DQLV_DYN',lat_lim=lat_lim).load())
    if latmean:
        raw_dtmse = dtmse_sel.sel(lev=slice(plim,None)).mean(['lat'])  # [lev, lon]
    else:
        raw_dtmse = dtmse_sel.sel(lev=slice(plim,None))
    mse_budget['dyn'] = raw_dtmse * 86400

    # dtmse_sel = (get_local_MSE_source(ds,'DDSE_CEF',lat_lim=lat_lim).load().copy() 
    #              + get_local_MSE_source(ds,'DQLV_CEF',lat_lim=lat_lim).load()) 
    # raw_dtmse = dtmse_sel.sel(lev=slice(plim,None)).mean(['lat'])  # [lev, lon]
    # mse_budget['cef'] = raw_dtmse * 86400

    # dtmse_sel = get_local_MSE_source(ds,'DDSE_GWD',lat_lim=lat_lim).load().copy() 
    # raw_dtmse = dtmse_sel.sel(lev=slice(plim,None)).mean(['lat'])  # [lev, lon]
    # mse_budget['gwd'] = raw_dtmse * 86400
    # resi = mse_budget['tendency'] - mse_budget['crmpbl'] - mse_budget['qr'] - mse_budget['dyn'] - mse_budget['gwd'] - mse_budget['cef']
    
    # resi = mse_budget['tendency'] - mse_budget['crmpbl'] - mse_budget['qr'] - mse_budget['dyn']
    # resi = mse_budget['tendency'] - mse_budget['crm'] - mse_budget['pbl'] - mse_budget['qr'] - mse_budget['dyn']
    # mse_budget['resi'] = resi

    return mse_budget

def get_integrated_MSE_tendency(ds, lat_lim=50, plim=None, latmean=False, iceflg=False):
    
    # [time, lev, lat, lon] or [time, lev, lon]
    local_tend = get_local_MSE_tendency(ds, lat_lim=lat_lim, latmean=latmean, iceflg=iceflg)

    # integration over pressure
    # int_mse = sum(local_mse * dp) / g 
    p = local_tend['lev'].values
    dp = np.zeros(len(p))

    dp[1:] = p[1:] - p[0:-1]
    dp[0] = dp[1]  # hPa
    g = 9.8

    # find the tropopause by T
    if plim is None:
        T = ds['T'].sel(lat=slice(-10, 10)).mean(dim=["time","lat","lon"]).values
        tp_index = np.argmin(T)
        if latmean:
            int_tend = (local_tend[:,tp_index:,:] * np.reshape(dp[tp_index:], (1, len(dp[tp_index:]), 1)) / g * 100).sum(dim='lev').squeeze()
        else:
            int_tend = (local_tend[:,tp_index:,:,:] * np.reshape(dp[tp_index:], (1, len(dp[tp_index:]),1, 1)) / g * 100).sum(dim='lev').squeeze()
    else:
        tp_index = np.argmin(np.abs(p - plim))
        if latmean:
            int_tend = (local_tend[:,tp_index:,:] * np.reshape(dp[tp_index:], (1, len(dp[tp_index:]), 1)) / g * 100).sum(dim='lev').squeeze()
        else:
            int_tend = (local_tend[:,tp_index:,:,:] * np.reshape(dp[tp_index:], (1, len(dp[tp_index:]), 1, 1)) / g * 100).sum(dim='lev').squeeze()
    # int_tend = (local_tend[:,tp_index:,:,:] * np.reshape(dp[tp_index:], (1, len(dp[tp_index:]), 1, 1)) / g * 100).sum(dim='lev').squeeze()

    return int_tend

def get_integrated_MSE_source(ds, varn, lat_lim=50, plim=None, latmean=False, iceflg=False):
    
    # [time, lev, lat, lon] or [time, lev, lon]
    local_src = get_local_MSE_source(ds, varn, lat_lim=lat_lim, latmean=latmean, iceflg=iceflg)

    # integration over pressure
    # int_mse = sum(local_mse * dp) / g 
    p = local_src['lev'].values
    dp = np.zeros(len(p))

    dp[1:] = p[1:] - p[0:-1]
    dp[0] = dp[1]  # hPa
    g = 9.8

    # find the tropopause by T
    if plim is None:
        T = ds['T'].sel(lat=slice(-10, 10)).mean(dim=["time","lat","lon"]).values
        tp_index = np.argmin(T)
        if latmean:
            int_src = (local_src[:,tp_index:,:] * np.reshape(dp[tp_index:], (1, len(dp[tp_index:]), 1)) / g * 100).sum(dim='lev').squeeze()
        else:
            int_src = (local_src[:,tp_index:,:,:] * np.reshape(dp[tp_index:], (1, len(dp[tp_index:]),1, 1)) / g * 100).sum(dim='lev').squeeze()
    else:
        tp_index = np.argmin(np.abs(p - plim))
        if latmean:
            int_src = (local_src[:,tp_index:,:] * np.reshape(dp[tp_index:], (1, len(dp[tp_index:]), 1)) / g * 100).sum(dim='lev').squeeze()
        else:
            int_src = (local_src[:,tp_index:,:,:] * np.reshape(dp[tp_index:], (1, len(dp[tp_index:]), 1, 1)) / g * 100).sum(dim='lev').squeeze()
    
    return int_src

def get_integrated_MSE_budget(ds, lat_lim=50, plim=None, latmean=False, iceflg=False):
    mse_budget = {}
    # get integrated MSE
    int_mse = get_integrated_MSE(ds, lat_lim=lat_lim, plim=plim, latmean=latmean, iceflg=iceflg).load()
    mse_budget['mse'] = int_mse

    # get integrated MSE tendency
    int_mse_tend = get_integrated_MSE_tendency(ds, lat_lim=lat_lim, plim=plim, latmean=latmean, iceflg=iceflg).load()
    mse_budget['tendency'] = int_mse_tend

    # get local MSE source: CRM
    dtmse_sel = (get_integrated_MSE_source(ds,'DDSE_CRM',lat_lim=lat_lim, latmean=latmean).load().copy()
                + get_integrated_MSE_source(ds,'DQLV_CRM',lat_lim=lat_lim, latmean=latmean).load().copy())
    # raw_dtmse = dtmse_sel.sel(lev=slice(plim,None))
    mse_budget['crm'] = dtmse_sel * 86400

    # get local MSE source: CRM_ALT
    dtmse_sel = (
                 get_integrated_MSE_source(ds,'DDSE_CRM_ALT',lat_lim=lat_lim, latmean=latmean).load().copy()
                 + get_integrated_MSE_source(ds,'DQLV_CRM_ALT',lat_lim=lat_lim, latmean=latmean).load().copy()
                 - get_integrated_MSE_source(ds,'DDSE_QRS',lat_lim=lat_lim, latmean=latmean).load().copy()
                 - get_integrated_MSE_source(ds,'DDSE_QRL',lat_lim=lat_lim, latmean=latmean).load().copy()
                )
    # raw_dtmse = dtmse_sel.sel(lev=slice(plim,None))
    mse_budget['crmalt'] = dtmse_sel * 86400

    # get local MSE source: PBL
    dtmse_sel = (get_integrated_MSE_source(ds,'DDSE_PBL',lat_lim=lat_lim, latmean=latmean).load().copy()
                + get_integrated_MSE_source(ds,'DQLV_PBL',lat_lim=lat_lim, latmean=latmean).load().copy())
    # raw_dtmse = dtmse_sel.sel(lev=slice(plim,None))
    mse_budget['pbl'] = dtmse_sel * 86400

    dtmse_sel = (get_integrated_MSE_source(ds,'DDSE_QRS',lat_lim=lat_lim, latmean=latmean).load().copy() 
                 + get_integrated_MSE_source(ds,'DDSE_QRL',lat_lim=lat_lim, latmean=latmean).load().copy())
    # raw_dtmse = dtmse_sel.sel(lev=slice(plim,None))
    mse_budget['qr'] = dtmse_sel * 86400

    dtmse_sel = (get_integrated_MSE_source(ds,'DDSE_DYN',lat_lim=lat_lim, latmean=latmean).load().copy() 
                 + get_integrated_MSE_source(ds,'DQLV_DYN',lat_lim=lat_lim, latmean=latmean).load().copy())
    # raw_dtmse = dtmse_sel.sel(lev=slice(plim,None))
    mse_budget['dyn'] = dtmse_sel * 86400

    return mse_budget

def get_integration(local_tend, ds):
    # integration over pressure
    # int_mse = sum(local_mse * dp) / g 
    p = local_tend['lev'].values
    dp = np.zeros(len(p))

    dp[1:] = p[1:] - p[0:-1]
    dp[0] = dp[1]  # hPa

    # find the tropopause by T

    T = ds['T'].sel(lat=slice(-10, 10)).mean(dim=["time","lat","lon"]).values
    tp_index = np.argmin(T)

    g = 9.8

    int_tend = (local_tend[:,tp_index:,:,:] * np.reshape(dp[tp_index:], (1, len(dp[tp_index:]), 1, 1)) / g * 100).sum(dim='lev').squeeze()

    return int_tend

# def get_LMSE_ano2_term(hp, Qp, ds):
#     # hp is MJO-filtered MSE anomaly
#     # Qp is MJO-filtered MSE source/tendency
#     # ds is the used 3D dataset
#     int_mse_ano2_term = get_integration(hp * Qp, ds)

#     return int_mse_ano2_term

# def get_GMSE_ano2_term(hp, Qp, ds):
#     # hp is MJO-filtered MSE anomaly
#     # Qp is MJO-filtered MSE source/tendency
#     # ds is the used 3D dataset

#     int_src = get_integration(Qp, ds)
#     int_mse = get_integration(hp, ds)
#     output = int_src * int_mse

#     return output

# def get_LMSE_budget_term(hp, Qp):
#     # hp[lev, lat, lon]; lev: only the troposphere
#     # Qp[lev, lat, lon]

#     # return variance production [lat, lon]

#     p = hp['lev'].values
#     dp = np.zeros(len(p))

#     dp[1:] = p[1:] - p[0:-1]
#     dp[0] = dp[1]  # hPa

#     g = 9.8

#     int_prod = (hp * Qp * xr.DataArray(dp, dims=['lev']) / g * 100).sum(dim='lev').squeeze()

#     return int_prod

# def get_GMSE_budget_term(hp, Qp):
#     # hp[lev, lat, lon]; lev: only the troposphere
#     # Qp[lev, lat, lon]

#     # return variance production [lat, lon]

#     p = hp['lev'].values
#     dp = np.zeros(len(p))

#     dp[1:] = p[1:] - p[0:-1]
#     dp[0] = dp[1]  # hPa

#     g = 9.8

#     int_hp = (hp * xr.DataArray(dp, dims=['lev']) / g * 100).sum(dim='lev').squeeze()
#     int_Qp = (Qp * xr.DataArray(dp, dims=['lev']) / g * 100).sum(dim='lev').squeeze()
    
#     return int_hp * int_Qp

def reorder_lon(u, lon0=181):

    # left shift the dataset 
    dim_lon = u.dims.index('lon')
    lon0_index = np.where(u.lon.values == lon0)[0]
    u_rolled_values = np.roll(u.values, -lon0_index, axis=dim_lon)

    lon_rolled = np.roll(u['lon'].values, -lon0_index)

    u_rolled = xr.DataArray(
        data= u_rolled_values,
        coords= u.coords,
        dims= u.dims,
    )

    u_rolled['lon'] = lon_rolled

    return u_rolled

# def MJO_regression(u, olr, ndays=None):
#     '''
#     This function is to get the regression coefficients at each grid point for 
#     each input variable field based on the reference OLR time series. 
    
#     ###############
#     input:
#     u: a variable field needed regression [time, lat, lon].
#     olr: outgoing longwave radiation field to find the reference OLR time series.
#     ndays: how many model days included in the calculation.
#     ###############
#     return:
#     regcoef: regression coefficients showing the linear relationship between u and olr
#     [lat, lon]

#     ###############
#     steps:
#     1. filter MJO-related signals for both u and olr.
#     2. calculate the time variance of OLR at each point.
#         and find the reference latitude with the largest variance.
#     3. create the long OLR reference time seriers. 
#     4. do linear regression for each point on the u field and get the coefficients.
#     '''

#     # get MJO-filtered OLR and u 
#     olr_flt = {}

#     for lat in olr['lat'].values:
#         olr_flt[lat] = get_MJO_signal(olr.sel(lat=lat).load().squeeze()).real

#     filtered_olr = xr.DataArray(list(olr_flt.values()), 
#                             coords=[list(olr_flt.keys()), olr['time'].values, olr['lon'].values], 
#                             dims=['lat','time','lon']).transpose('time', 'lat', 'lon')
    
#     u_flt = {}

#     for lat in olr['lat'].values:
#         u_flt[lat] = get_MJO_signal(u.sel(lat=lat).load().squeeze()).real

#     filtered_u = xr.DataArray(list(u_flt.values()), 
#                             coords=[list(u_flt.keys()), u['time'].values, u['lon'].values], 
#                             dims=['lat','time','lon']).transpose('time', 'lat', 'lon')
    
#     # find the reference latitude
#     if ndays==None:
#         ndays = len(olr['time'].values)

#     kt_damped = int(np.rint(ndays/4))
#     varmax_lat_index = filtered_olr[kt_damped:-kt_damped,:,:].var(dim='time').mean(dim='lon').argmax()
#     varmax_lat = olr['lat'][varmax_lat_index].values

#     # get the reference OLR time series.
#     # the default basis longitude is 181
#     olr_rolled = reorder_lon(filtered_olr[:, varmax_lat_index, :].squeeze())
#     olrref = np.concatenate([olr_rolled.sel(lon=lon_value).values for lon_value in olr_rolled['lon'].values])
#     del olr_rolled
#     del filtered_olr
#     del olr_flt

#     regcoef = xr.DataArray(
#         data=np.zeros(filtered_u[0,:,:].squeeze().shape),
#         coords={
#             'lat': filtered_u['lat'].values,
#             'lon': filtered_u['lon'].values,
#         },
#         dims=['lat','lon'],
#     )

#     for lon in filtered_u['lon'].values:
#         u_rolled = reorder_lon(filtered_u, lon0=lon)
#         u_yaixs = np.concatenate([u_rolled.sel(lon=lon_value).values for lon_value in u_rolled['lon'].values])

#         lon_index = np.where(filtered_u['lon'].values == lon)[0]

#         coef = np.polyfit(olrref, u_yaixs, deg=1)
#         regcoef[:,lon_index] = np.reshape(coef[0,:], (len(coef[0,:]),1))
#         del u_rolled

#     return regcoef

# from numba import jit
# import multiprocessing as mp

# # @jit(nopython=True)
# def polyfit_jit(olr_ref_array, u_array, lat, lon, coefficients):
#     u_lat_lon = u_array[:, lat, lon]
#     slope, intercept = np.polyfit(olr_ref_array, u_lat_lon, 1)
#     coefficients[lat, lon, 0] = slope
#     coefficients[lat, lon, 1] = intercept

# def linear_regression_parallel(u, olr_ref):
#     # u can be 2D, 3D, or 4D
#     # olr_ref is 1D
#     # make sure the first dimension is time

#     u_array = np.asarray(u)  # Convert u to a NumPy array
#     olr_ref_array = np.asarray(olr_ref)  # Convert olr_ref to a NumPy array

#     # Get the number of dimensions in u_array
#     num_dimensions = len(u_array.shape)

#     if num_dimensions == 3:  # [time, lat, lon]
#         time_dim, lat_dim, lon_dim = u_array.shape
#         coefficients = np.zeros((lat_dim, lon_dim, 2))  # 2 for slope and intercept

#         pool = mp.Pool(mp.cpu_count())

#         for lat in range(lat_dim):
#             for lon in range(lon_dim):
#                 pool.apply_async(polyfit_jit, args=(olr_ref_array, u_array, lat, lon, coefficients))
        
#         pool.close()
#         pool.join()

#     elif num_dimensions == 2:  # [time, lon]
#         time_dim, lon_dim = u_array.shape
#         coefficients = np.zeros((lon_dim, 2))  # 2 for slope and intercept

#         for lon in range(lon_dim):
#             u_lon = u_array[:, lon]
#             slope, intercept = np.polyfit(olr_ref_array, u_lon, 1)

#             coefficients[lon, 0] = slope
#             coefficients[lon, 1] = intercept

#     elif num_dimensions == 4:  # [time, lev, lat, lon]
#         time_dim, lev_dim, lat_dim, lon_dim = u_array.shape
#         coefficients = np.zeros((lev_dim, lat_dim, lon_dim, 2))  # 2 for slope and intercept

#         for lev in range(lev_dim):
#             for lat in range(lat_dim):
#                 for lon in range(lon_dim):
#                     u_lev_lat_lon = u_array[:, lev, lat, lon]
#                     slope, intercept = np.polyfit(olr_ref_array, u_lev_lat_lon, 1)

#                     coefficients[lev, lat, lon, 0] = slope
#                     coefficients[lev, lat, lon, 1] = intercept

#     return coefficients    

def linear_regression_old(u, olr_ref):
    # u can be 2D, 3D, or 4D
    # olr_ref is 1D
    # make sure the first dimension is time

    u_array = np.asarray(u)  # Convert u to a NumPy array
    olr_ref_array = np.asarray(olr_ref)  # Convert olr_ref to a NumPy array

    # Get the number of dimensions in u_array
    num_dimensions = len(u_array.shape)

    if num_dimensions == 3:  # [time, lat, lon]
        time_dim, lat_dim, lon_dim = u_array.shape
        coefficients = np.zeros((lat_dim, lon_dim, 2))  # 2 for slope and intercept

        for lat in range(lat_dim):
            for lon in range(lon_dim):
                u_lat_lon = u_array[:, lat, lon]

                slope, intercept = np.polyfit(olr_ref_array, u_lat_lon, 1)

                coefficients[lat, lon, 0] = slope
                coefficients[lat, lon, 1] = intercept

    elif num_dimensions == 2:  # [time, lon]
        time_dim, lon_dim = u_array.shape
        coefficients = np.zeros((lon_dim, 2))  # 2 for slope and intercept

        for lon in range(lon_dim):
            u_lon = u_array[:, lon]
            slope, intercept = np.polyfit(olr_ref_array, u_lon, 1)

            coefficients[lon, 0] = slope
            coefficients[lon, 1] = intercept

    elif num_dimensions == 4:  # [time, lev, lat, lon]
        time_dim, lev_dim, lat_dim, lon_dim = u_array.shape
        coefficients = np.zeros((lev_dim, lat_dim, lon_dim, 2))  # 2 for slope and intercept

        for lev in range(lev_dim):
            for lat in range(lat_dim):
                for lon in range(lon_dim):
                    u_lev_lat_lon = u_array[:, lev, lat, lon]
                    slope, intercept = np.polyfit(olr_ref_array, u_lev_lat_lon, 1)

                    coefficients[lev, lat, lon, 0] = slope
                    coefficients[lev, lat, lon, 1] = intercept

    return coefficients

# new version only returns slopes
def linear_regression(u, olr_ref):
    # u can be 2D, 3D, or 4D
    # olr_ref is 1D
    # make sure the first dimension is time

    u_array = np.asarray(u)  # Convert u to a NumPy array
    olr_ref_array = np.asarray(olr_ref)  # Convert olr_ref to a NumPy array

    # Get the number of dimensions in u_array
    num_dimensions = len(u_array.shape)

    if num_dimensions == 3:  # [time, lat, lon]
        _ , lat_dim, lon_dim = u_array.shape
        coefficients = np.zeros((lat_dim, lon_dim))  # 2 for slope and intercept

        for lat in range(lat_dim):
            for lon in range(lon_dim):
                u_lat_lon = u_array[:, lat, lon]

                slope, _ = np.polyfit(olr_ref_array, u_lat_lon, 1)

                coefficients[lat, lon] = slope

    elif num_dimensions == 2:  # [time, lon]
        _ , lon_dim = u_array.shape
        coefficients = np.zeros((lon_dim))  # 2 for slope and intercept

        for lon in range(lon_dim):
            u_lon = u_array[:, lon]
            slope, _ = np.polyfit(olr_ref_array, u_lon, 1)

            coefficients[lon] = slope

    elif num_dimensions == 4:  # [time, lev, lat, lon]
        _ , lev_dim, lat_dim, lon_dim = u_array.shape
        coefficients = np.zeros((lev_dim, lat_dim, lon_dim))  # 2 for slope and intercept

        for lev in range(lev_dim):
            for lat in range(lat_dim):
                for lon in range(lon_dim):
                    u_lev_lat_lon = u_array[:, lev, lat, lon]
                    slope, _ = np.polyfit(olr_ref_array, u_lev_lat_lon, 1)

                    coefficients[lev, lat, lon] = slope

    return coefficients

def concat_olr(olr_ref):
    # olr_ref is a 2D array [time, lon]. 
    # return a 1D array [time*lon] for regression
    olr_ref_array = np.asarray(olr_ref)  # Convert olr_ref to a NumPy array
    return olr_ref_array.T.reshape(-1)

def global_linear_reg(u, olr_ref):
    # u is a field to be regressed
    # olr_ref is the reference OLR time series [time, lon]
    # step 1: concatenate the reference OLR into a long time series
    olr_long_values = concat_olr(olr_ref) # [time*lon]

    # step 2: concatenate the u into a long time series
    kt, klon = olr_ref.shape  # length of time and longitude
    ushape = u.shape

    if len(ushape) == 2:
        u_long_values = np.empty(kt*klon,klon)
        # shift longitude to make each xi at the center
        dim_lon = u.dims.index('lon')
        for lon0_index in range(klon):
            u_long_values[lon0_index*kt:(lon0_index+1)*kt,:] = np.roll(u.values, 90-lon0_index, axis=dim_lon)

    elif len(ushape) == 3:
        u_long_values = np.empty((kt*klon, ushape[1], klon))
        # shift longitude to make each xi at the center
        dim_lon = u.dims.index('lon')
        for lon0_index in range(klon):
            u_long_values[lon0_index*kt:(lon0_index+1)*kt,:,:] = np.roll(u.values, 90-lon0_index, axis=dim_lon)

    elif len(ushape) == 4:
        u_long_values = np.empty((kt*klon, ushape[1], ushape[2], klon))   
        # shift longitude to make each xi at the center
        dim_lon = u.dims.index('lon')
        for lon0_index in range(klon):
            u_long_values[lon0_index*kt:(lon0_index+1)*kt,:,:,:] = np.roll(u.values, 90-lon0_index, axis=dim_lon)

    # step 3: do linear regression
    # regcoef = linear_regression_parallel(u_long_values, olr_long_values)
    regcoef = linear_regression(u_long_values, olr_long_values)

    return regcoef

def np_2_xr(values, xrarray):
    # values is a numpy array
    # xrarray is an xarray DataArray
    # return an xarray DataArray with the same coordinates as xrarray
    return xr.DataArray(
        data=values,
        coords=xrarray.coords,
        dims=xrarray.dims,
    )

def get_reg_local_MSE_budegt(ds, olr_flt, lat_lim=5, latmean=True, plim=120):

    coef = {}

    mse_budget = get_local_MSE_budget(ds, lat_lim=lat_lim, plim=plim, latmean=latmean)

    olrstd = olr_flt.std().values

    # get local MSE
    coef_h = global_linear_reg(mse_budget['mse'], olr_flt)
    coef['MSE'] = - 2 * olrstd * coef_h

    # get local MSE tendency
    coef_dth = global_linear_reg(mse_budget['tendency'], olr_flt)
    coef['TEND'] = - 2 * olrstd * coef_dth

    # get local MSE source: CRM + PBL
    coef_dth = global_linear_reg(mse_budget['crmalt'], olr_flt)
    coef['CRMALT'] = - 2 * olrstd * coef_dth

    # get local MSE source: CRM + PBL
    coef_dth = global_linear_reg(mse_budget['crm'], olr_flt)
    coef['CRM'] = - 2 * olrstd * coef_dth
    # get local MSE source: CRM + PBL
    coef_dth = global_linear_reg(mse_budget['pbl'], olr_flt)
    coef['PBL'] = - 2 * olrstd * coef_dth

    # get local MSE source: QR
    coef_dth = global_linear_reg(mse_budget['qr'], olr_flt)
    coef['RAD'] = - 2 * olrstd * coef_dth

    # get local MSE source: DYN
    coef_dth = global_linear_reg(mse_budget['dyn'], olr_flt)
    coef['DYN'] = - 2 * olrstd * coef_dth


    return coef

def get_reg_integrated_MSE_budegt(ds, olr_flt, lat_lim=5, latmean=False, plim=None):

    coef = {}

    mse_budget = get_integrated_MSE_budget(ds, lat_lim=lat_lim, plim=plim, latmean=latmean)

    olrstd = olr_flt.std().values

    # get local MSE
    coef_h = global_linear_reg(mse_budget['mse'], olr_flt)
    coef['MSE'] = - 2 * olrstd * coef_h

    # get local MSE tendency
    coef_dth = global_linear_reg(mse_budget['tendency'], olr_flt)
    coef['TEND'] = - 2 * olrstd * coef_dth

    # get local MSE source: CRM + PBL
    coef_dth = global_linear_reg(mse_budget['crmalt'], olr_flt)
    coef['CRMALT'] = - 2 * olrstd * coef_dth

    # get local MSE source: CRM + PBL
    coef_dth = global_linear_reg(mse_budget['crm'], olr_flt)
    coef['CRM'] = - 2 * olrstd * coef_dth
    # get local MSE source: CRM + PBL
    coef_dth = global_linear_reg(mse_budget['pbl'], olr_flt)
    coef['PBL'] = - 2 * olrstd * coef_dth

    # get local MSE source: QR
    coef_dth = global_linear_reg(mse_budget['qr'], olr_flt)
    coef['RAD'] = - 2 * olrstd * coef_dth

    # get local MSE source: DYN
    coef_dth = global_linear_reg(mse_budget['dyn'], olr_flt)
    coef['DYN'] = - 2 * olrstd * coef_dth

    return coef

def get_local_MSE_budget_composite(mse_budget, olrmin):
    comp = {}

    for key in mse_budget.keys():
        mse = mse_budget[key]
        mse_ano = mse - mse.mean(dim='lon')
        mse_sft = mse_ano.copy()

        for i in range(olrmin.size):
            mse_sft[i,:,:] = np.roll(mse_ano[i, :, :], shift=90-olrmin[i], axis=-1)
        
        comp[key] = mse_sft.mean(dim='time')
        print(key)

    return comp


# draw Hovmoller diagrams
def get_hovmoller(case_dir='control', daya=0, dayb=1500, figsize=(7,8)):

    dirn = '/pscratch/sd/l/linyaoly/MJO_E3SM/regridded_data/'

    dspw = xr.open_dataset(dirn+'analysis/hovmoller_pw_'+case_dir+'.nc')
    pwavg = dspw['pw_raw']

    dsprep = xr.open_dataset(dirn+'analysis/hovmoller_prep_'+case_dir+'.nc')
    prepavg = dsprep['prep_raw']
    fig, ax = plt.subplots(figsize=figsize)
    ftsize = 32
    plt.rcParams.update({'font.size': ftsize})
    # Plot u850 as a colored background

    data = pwavg[daya:dayb,:] - pwavg.mean(dim='time')

    if case_dir == 'control':
        vmin = -14
        vmax = 14
    elif case_dir == 'FIX_QRT':
        vmin = -14
        vmax = 14
    else:
        vmin = -14
        vmax = 14

    im = ax.contourf(pwavg[daya:dayb,:].coords['lon'], pwavg[daya:dayb,:].coords['time']-pwavg[daya,:].coords['time'], data, cmap='RdBu_r', levels=np.linspace(vmin,vmax,15)) # np.linspace(-10,10,51))

    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.04)

    # Add contours for negative OLR
    # contour_levels = [-200, -150, -100, -50]  # customize this as needed
    data = (prepavg[daya:dayb,:]-prepavg.mean(dim='time'))
    olr_contours = ax.contour(prepavg[daya:dayb,:].coords['lon'], prepavg[daya:dayb,:].coords['time']-prepavg[daya,:].coords['time'], data.where(data> 0), colors='k', levels=4, linewidths=1)
    ax.tick_params(axis='both', labelsize=ftsize, which='major', length=10, width=1.5)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(0, 361, 90))
    # Show the plot
    plt.savefig('./plots/hovmoller_'+case_dir+'_unfiltered_pwcolor_prepline.png', bbox_inches='tight')

    dspw = xr.open_dataset(dirn+'analysis/hovmoller_pw_'+case_dir+'.nc')
    pwavg = dspw['pw_flt']

    dsprep = xr.open_dataset(dirn+'analysis/hovmoller_prep_'+case_dir+'.nc')
    prepavg = dsprep['prep_flt']
    fig, ax = plt.subplots(figsize=(7, 8))
    ftsize = 32
    plt.rcParams.update({'font.size': ftsize})
    # Plot u850 as a colored background

    data = pwavg[daya:dayb,:] 
    if case_dir == 'control':
        vmin = -7
        vmax = 7
        ticks = [-6, -4, -2, 0, 2, 4, 6]
    elif case_dir == 'FIX_QRT':
        vmin = -7
        vmax = 7
        ticks = [-6, -4, -2, 0, 2, 4, 6]
    else:
        vmin = -7
        vmax = 7
        ticks = [-6, -4, -2, 0, 2, 4, 6]

    im = ax.contourf(pwavg[daya:dayb,:].coords['lon'], pwavg[daya:dayb,:].coords['time']-pwavg[daya,:].coords['time'], data, cmap='RdBu_r', levels=np.linspace(vmin,vmax,15)) # np.linspace(-10,10,51))

    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.04, ticks=ticks)

    # Add contours for negative OLR
    # contour_levels = [-200, -150, -100, -50]  # customize this as needed
    data = prepavg[daya:dayb,:]
    olr_contours = ax.contour(prepavg[daya:dayb,:].coords['lon'], prepavg[daya:dayb,:].coords['time']-prepavg[daya,:].coords['time'], data.where(data> 0), colors='k', levels=4, linewidths=1)
    ax.tick_params(axis='both', labelsize=ftsize, which='major', length=10, width=1.5)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(0, 361, 90))
    # Show the plot
    plt.savefig('./plots/hovmoller_'+case_dir+'_mjofiltered_pwcolor_prepline.png', bbox_inches='tight')


    # OLR and U850
    dspw = xr.open_dataset(dirn+'analysis/hovmoller_u850_'+case_dir+'.nc')
    pwavg = dspw['u850_raw']

    dsprep = xr.open_dataset(dirn+'analysis/hovmoller_olr_'+case_dir+'.nc')
    prepavg = dsprep['olr_raw']
    fig, ax = plt.subplots(figsize=(7, 8))
    ftsize = 32
    plt.rcParams.update({'font.size': ftsize})
    # Plot u850 as a colored background

    data = pwavg[daya:dayb,:] - pwavg.mean(dim='time')
    if case_dir == 'control':
        vmin = -14
        vmax = 14
    elif case_dir == 'FIX_QRT':
        vmin = -14
        vmax = 14
    else:
        vmin = -14
        vmax = 14

    im = ax.contourf(pwavg[daya:dayb,:].coords['lon'], pwavg[daya:dayb,:].coords['time']-pwavg[daya,:].coords['time'], data, cmap='RdBu_r', levels=np.linspace(vmin,vmax,15)) # np.linspace(-10,10,51))

    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.04)

    # Add contours for negative OLR
    # contour_levels = [-200, -150, -100, -50]  # customize this as needed
    data = (prepavg[daya:dayb,:]-prepavg.mean(dim='time'))
    olr_contours = ax.contour(prepavg[daya:dayb,:].coords['lon'], prepavg[daya:dayb,:].coords['time']-prepavg[daya,:].coords['time'], data.where(data< 0), colors='k', levels=3, linewidths=1)
    ax.tick_params(axis='both', labelsize=ftsize, which='major', length=10, width=1.5)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(0, 361, 90))
    # Show the plot
    plt.savefig('./plots/hovmoller_'+case_dir+'_unfiltered_u850color_olrline.png', bbox_inches='tight')

    pwavg = dspw['u850_flt']

    prepavg = dsprep['olr_flt']
    fig, ax = plt.subplots(figsize=(7, 8))
    ftsize = 32
    plt.rcParams.update({'font.size': ftsize})
    # Plot u850 as a colored background

    data = pwavg[daya:dayb,:] 
    if case_dir == 'control':
        vmin = -14
        vmax = 14
        ticks = [-12, -8, -4, 0, 4, 8, 12]
    elif case_dir == 'FIX_QRT':
        vmin = -14
        vmax = 14
        ticks = [-12, -8, -4, 0, 4, 8, 12]
    else:
        vmin = -14
        vmax = 14
        ticks = [-12, -8, -4, 0, 4, 8, 12]

    im = ax.contourf(pwavg[daya:dayb,:].coords['lon'], pwavg[daya:dayb,:].coords['time']-pwavg[daya,:].coords['time'], data, cmap='RdBu_r', levels=np.linspace(vmin,vmax,15)) # np.linspace(-10,10,51))

    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.04, ticks=ticks)

    # Add contours for negative OLR
    # contour_levels = [-200, -150, -100, -50]  # customize this as needed
    data = prepavg[daya:dayb,:]
    olr_contours = ax.contour(prepavg[daya:dayb,:].coords['lon'], prepavg[daya:dayb,:].coords['time']-prepavg[daya,:].coords['time'], data.where(data< 0), colors='k', levels=3, linewidths=1)
    ax.tick_params(axis='both', labelsize=ftsize, which='major', length=10, width=1.5)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(0, 361, 90))
    # Show the plot
    plt.savefig('./plots/hovmoller_'+case_dir+'_mjofiltered_u850color_olrline.png', bbox_inches='tight')

def get_hovmoller_olr(case_dir='control', daya=0, dayb=1500, figsize=(7,8)):

    dirn = '/pscratch/sd/l/linyaoly/MJO_E3SM/regridded_data/'

    # OLR 
    dsprep = xr.open_dataset(dirn+'analysis/hovmoller_olr_'+case_dir+'.nc')
    
    prepavg = dsprep['olr_raw']
    fig, ax = plt.subplots(figsize=(7, 8))
    ftsize = 32
    plt.rcParams.update({'font.size': ftsize})
    # Plot u850 as a colored background

    data0 = (prepavg[daya:dayb,:]-prepavg.mean(dim='time'))
    data = data0 - data0.mean(dim='lon')
    
    vmin = -75
    vmax = 75

    im = ax.contourf(prepavg[daya:dayb,:].coords['lon'], prepavg[daya:dayb,:].coords['time']-prepavg[daya,:].coords['time'], data, cmap='RdBu_r', levels=np.linspace(vmin,vmax,31)) # np.linspace(-10,10,51))

    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.04)

    ax.tick_params(axis='both', labelsize=ftsize, which='major', length=10, width=1.5)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(0, 361, 90))
    # Show the plot
    plt.savefig('./plots/hovmoller_'+case_dir+'_unfiltered_olr.png', bbox_inches='tight')

    # prepavg = dsprep['olr_flt']
    # fig, ax = plt.subplots(figsize=(7, 8))
    # ftsize = 32
    # plt.rcParams.update({'font.size': ftsize})
    # # Plot u850 as a colored background

    # data = prepavg[daya:dayb,:] 

    # vmin = -35
    # vmax = 35

    # im = ax.contourf(prepavg[daya:dayb,:].coords['lon'], prepavg[daya:dayb,:].coords['time']-prepavg[daya,:].coords['time'], data, cmap='RdBu_r', levels=np.linspace(vmin,vmax,15)) # np.linspace(-10,10,51))

    # # Add a colorbar
    # cbar = plt.colorbar(im, ax=ax, pad=0.04)
    # ax.tick_params(axis='both', labelsize=ftsize, which='major', length=10, width=1.5)
    # ax.invert_yaxis()
    # ax.set_xticks(np.arange(0, 361, 90))
    # # Show the plot
    # plt.savefig('./plots/hovmoller_'+case_dir+'_mjofiltered_olr.png', bbox_inches='tight')




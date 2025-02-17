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

def hovmoller_power(data, segsize=96, spd=1):
    """
    data: [time, layer, memory, lon]
    Method
    ------------------
        4. Detrend the segments (remove linear trend).
        5. Apply taper in time dimension of windows (aka segments).
        6. Fourier transform
        7. calculate the power.
        8. average over all segments
        9. sum the power over all latitudes.
        
    """

    segsize = spd * segsize  # how many time steps included.
    # # 4. 
    # # Apply the detrend function to each segment

    x_roll = xr.DataArray(
        data=np.asarray(data),
        dims=("time","lat","segments","lon"),
    )

    original_dims = x_roll.dims
    x_roll_sym_asym = x_roll.chunk({'time': 1, 'lat': 'auto', 'lon': 'auto', 'segments': -1})

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
    taper = split_hann_taper(seg_size=segsize, fraction=0.2)

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
    

    # 7. [time, channel, wavenumber, frequency]
    # reorder coef matrix according to ncl script
    fft_reorder = Hayashi(fft_sym_asym, segsize/spd)  # k is from -wavenumber to wavenumber; f is from -frequency to frequency
    # fft_asym_reorder = Hayashi(fft_asym, segsize/spd)
    
    # 8. average over all segments [wavenumber, frequency]
    zsym = 2.0 * fft_reorder.mean(dim='time').squeeze()
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


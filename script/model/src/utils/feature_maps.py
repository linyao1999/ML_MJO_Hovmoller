import numpy as np
import math
import torch
from scipy import special
import xarray as xr
from concurrent.futures import ProcessPoolExecutor
import re

# ========================================================================
# spectral analysis for feature maps 
# ========================================================================
def get_um(u, lat_lim=20, c=51):
    lat = np.arange(lat_lim,-lat_lim-2,-2)
    max_m = len(lat)
    beta= 2.28e-11  # variation of coriolis parameter with latitude
    L = np.sqrt(c / beta)  # horizontal scale (m)
    y = lat * 110 * 1000 / L # dimensionless
    phi = []

    for i in np.arange(max_m):
        p = special.hermite(i)
        Hm = p(y)
        phim = np.exp(- y**2 / 2) * Hm / np.sqrt((2**i) * np.sqrt(np.pi) * math.factorial(i))

        if len(u.shape)==4:
            phi.append(np.reshape(phim, (1, 1, len(y), 1)))
        elif len(u.shape)==3:
            phi.append(np.reshape(phim, (1, len(y), 1)))
        else:
            print('wrong input shape!')
            exit()

    # projection coefficients
    if len(u.shape)==4:
        um = np.zeros((u.shape[0], u.shape[1], max_m, u.shape[-1]))

        dy = (lat[0] - lat[1]) * 110 * 1000 / L 

        for i in range(max_m):
            um0 = np.sum(u * phi[i] * dy, axis=2, keepdims=True)  # (time,channel, 1, lon)
            um[:,:,i,None,:] = um0
    elif len(u.shape)==3:
        um = np.zeros((u.shape[0], max_m, u.shape[-1]))

        dy = (lat[0] - lat[1]) * 110 * 1000 / L 

        for i in range(max_m):
            um0 = np.sum(u * phi[i] * dy, axis=1, keepdims=True)
            um[:,i,None,:] = um0

    return um

def get_hid_power_norm_one(
    fn,
    lat_lim = 20,
    dataflg = 'flt',
    c = 51,
    hidden_layer=[1,],
    after_relu=False,
    ):

    try:
        features = torch.load(fn, weights_only=True, map_location='cpu')
    except Exception as e:
        print(f"Failed to load {fn}: {e}")
        return

    # features = torch.load(fn,weights_only=True, map_location='cpu')
    layer_names = list(features.keys())
    # print(layer_names)

    hid = []
    for hidden in hidden_layer:
        layer_name = layer_names[hidden]
        # print(f'analyze {layer_name}')
        data = features[layer_name].numpy() # [time, c, lat, lon]
        # print(data.shape)
        hid.append(data)
    hid = np.concatenate(hid, axis=1) # [time, c, lat, lon]

    # get the hidden layers after ReLU
    if after_relu:
        hid = np.maximum(hid, 0)

    hid_m = get_um(hid, lat_lim=lat_lim, c=c) # [time, c, max_m, lon]

    # zonal fft
    hid_mk = np.fft.rfft(hid_m, axis=-1) # [time, c, max_m, lon//2+1]
    hid_mk_power = np.abs(hid_mk)**2

    # normalize the power; excluding the zonal zero frequency
    hid_mk_power[:,:,:,0] = 0.0
    hid_mk_power_norm = hid_mk_power / np.sum(hid_mk_power, axis=(-1,-2), keepdims=True)
    power_norm = np.mean(hid_mk_power_norm, axis=0) # [c, max_m, lon//2+1]

    # save the output
    out_ds = xr.Dataset(
        {
            'power_norm': (('c', 'm', 'k'), power_norm),
        },
        coords={
            'c': np.arange(power_norm.shape[0]),
            'm': np.arange(power_norm.shape[1]),
            'k': np.arange(power_norm.shape[2]),
        },
    )
    if after_relu:
        reluflg = '_relu'
    else:
        reluflg = ''
    
    hid_str = '_'.join([str(x) for x in hidden_layer])
    fn = fn.replace('_feature_maps.pt', f'_power_norm_ch{hid_str}_c{c}{reluflg}.nc')

    out_ds.to_netcdf(fn, mode='w')
    print(f'Saved to {fn}')
    print('Done!')

def get_hid_power_norm_ensemble(
    config,
    c=51,
    exp_num_list=np.arange(1,101),
    hidden_layer=[-1,],
    after_relu=False,
    ):  

    feature_path = config['prediction_save_path'].replace(".nc", "_feature_maps.pt")
    fn_list = []
    for exp_num in exp_num_list:
        fn = re.sub(r"exp\d+/", f"exp{exp_num}/", feature_path)
        fn_list.append(fn)
        
    # print(f'fn list: {fn_list}')
    lat_lim = config['data']['lat_range']
    dataflg = config['data']['dataflg']

    with ProcessPoolExecutor(max_workers=8) as executor:
        # Submit jobs without capturing results
        futures = [
            executor.submit(get_hid_power_norm_one, fn, lat_lim, dataflg, c, hidden_layer, after_relu)
            for fn in fn_list
        ]

    print('feature map analysis finished!')

# =========================================================================
# spectral analysis for kernels
# =========================================================================
def get_kernel_power_mk_one(
    config,
    exp_num,
    hidden_layer=['cnn.network.0',],
    ):

    model_path = re.sub(r"exp\d+/", f"exp{exp_num}/", config['model_save_path'])
    model_dict = torch.load(model_path, map_location='cpu', weights_only=True)  
    kernels = []
    norm_powers = []
    norm_power_means = []
    for hidden in hidden_layer:
        if hidden in model_dict:
            tmp = model_dict[f'{hidden}.weight'].numpy()  # [output_channel, input_channel, height, width]
            kernels.append(tmp)

            # Get kernel size dynamically
            kernel_h, kernel_w = tmp.shape[-2:]
            # Calculate the power spectrum of the kernel
            power_spectrum = np.abs(np.fft.fft2(tmp, axes=(-2, -1)))**2
            total_power = np.sum(power_spectrum, axis=(-2, -1), keepdims=True)
            norm_power = power_spectrum / total_power  # [output_channel, input_channel, ky, kx]
            norm_power_mean = np.mean(norm_power, axis=(0, 1))  # [ky, kx]
            norm_powers.append(norm_power)
            norm_power_means.append(norm_power_mean)
        else:
            print(f'Warning: {hidden} not found in the model dictionary!')

    # Get kernel size from last loaded kernel
    if kernels:
        kernel_h, kernel_w = kernels[-1].shape[-2:]
        ky = np.fft.fftfreq(kernel_h)
        kx = np.fft.fftfreq(kernel_w)
    else:
        print("No kernels found! Skipping save.")
        return

    feature_path = config['prediction_save_path'].replace(".nc","_kernels.npz")
    fn = re.sub(r"exp\d+/", f"exp{exp_num}/", feature_path)
    np.savez(fn, 
             norm_powers=norm_powers,  # list of arrays, will need allow_pickle=True when loading
             norm_power_means=norm_power_means, 
             kx=kx, 
             ky=ky,
             kernels=kernels)
    print(f"Saved kernel spectra to: {fn}\nNote: Load with np.load(..., allow_pickle=True)")

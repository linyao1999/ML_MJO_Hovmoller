import glob
import xarray as xr
import os

datadir = '/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/TB/'
good_files = []
bad_files = []

files = sorted(glob.glob(os.path.join(datadir, "*.nc")))

for f in files:
    try:
        ds = xr.open_dataset(f, engine="netcdf4", backend_kwargs={"mask_and_scale": False})
        ds.close()
        good_files.append(f)
    except Exception as e:
        print("❌ BAD:", f)
        bad_files.append(f)

print("\nGOOD:", len(good_files))
print("BAD :", len(bad_files))

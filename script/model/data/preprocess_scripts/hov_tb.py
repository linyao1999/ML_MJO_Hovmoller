#!/usr/bin/env python3
import os
import glob
import xarray as xr
from multiprocessing import Pool, cpu_count

# salloc --nodes 1 --qos interactive --time 04:00:00 --constraint cpu --account=m4287

datadir = "/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/TB/"
outdir  = "/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/TB_monthly/"
varname = "irwin_cdr"
lat_min, lat_max = -10.0, 10.0

os.makedirs(outdir, exist_ok=True)
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

def preprocess(ds):
    da = ds[varname].sel(lat=slice(lat_min, lat_max)).mean("lat")
    return da.to_dataset(name="tb")

def process_one_month(args):
    year, month = args
    mm = f"{month:02d}"

    outfile = os.path.join(outdir, f"TB_{year}_{mm}.nc")

    if os.path.exists(outfile):
        print(f"[INFO] Skipping existing file {outfile}")
        return
    
    pattern = os.path.join(datadir, f"GRIDSAT-B1.{year}.{mm}.*.nc")
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        print(f"[WARN] No files for {year}-{mm}")
        return

    print(f"[INFO] Processing {year}-{mm}: {len(files)} files")

    ds = xr.open_mfdataset(
        files,
        combine="nested",
        concat_dim="time",
        engine="h5netcdf",
        preprocess=preprocess,
        chunks={"time": 128},
        parallel=False,   # multiprocessing handles parallelism
    )

    
    encoding = {"tb": {"dtype": "float32", "zlib": True, "complevel": 4}}

    ds.to_netcdf(outfile, encoding=encoding)
    print(f"[INFO] Saved {outfile}")

def main():
    start_year = 1999
    end_year   = 2023

    jobs = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            jobs.append((y, m))

    # How many CPUs do we have?
    ncpu = cpu_count()
    print(f"[INFO] Detected {ncpu} CPUs — launching parallel workers")

    with Pool(processes=32) as pool:
        pool.map(process_one_month, jobs)

    print("[INFO] All months processed.")

if __name__ == "__main__":
    main()

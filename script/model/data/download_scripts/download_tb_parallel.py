import os
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# salloc --nodes 1 --qos interactive --time 04:00:00 --constraint cpu --account=m4287

# --------------------------
# USER SETTINGS
# --------------------------
start_year = 2023
end_year = 2024
outdir = "/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/TB"
os.makedirs(outdir, exist_ok=True)

hours = ["00", "03", "06", "09", "12", "15", "18", "21"]

# --------------------------
# Generate the list of files to download
# --------------------------
start_date = datetime(start_year, 1, 1)
end_date = datetime(end_year, 12, 31)

download_tasks = []

t = start_date
while t <= end_date:
    yyyy = t.strftime("%Y")
    mm = t.strftime("%m")
    dd = t.strftime("%d")

    for hh in hours:
        filename = f"GRIDSAT-B1.{yyyy}.{mm}.{dd}.{hh}.v02r01.nc"
        url = f"https://www.ncei.noaa.gov/thredds/fileServer/cdr/gridsat/{yyyy}/{filename}"
        outfile = os.path.join(outdir, filename)
        download_tasks.append((url, outfile))

    t += timedelta(days=1)

print(f"Total files to download: {len(download_tasks)}")

# --------------------------
# Download function
# --------------------------
def download_file(url, outfile):
    if os.path.exists(outfile):
        return f"Exists: {os.path.basename(outfile)}"

    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(outfile, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return f"Downloaded: {os.path.basename(outfile)}"
    else:
        return f"Missing: {url}"

# --------------------------
# Parallel execution
# --------------------------
MAX_THREADS = 64   # You can increase up to 32 or 64 on Perlmutter

with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    futures = [executor.submit(download_file, url, outfile)
               for url, outfile in download_tasks]

    for future in as_completed(futures):
        print(future.result())

print("DONE")

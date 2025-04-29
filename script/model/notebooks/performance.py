import sys
from pathlib import Path
src_utils_path = Path("../src/utils")
sys.path.append(str(src_utils_path))
import metrics as mjo
import numpy as np

import optuna
import os

# Example: salloc --nodes=1 --qos=interactive --time=04:00:00 --constraint=cpu --ntasks=8 --cpus-per-task=8 --account=dasrepo

lat_range = 10
lead = 35
memory_last = 29

for dataflg in ['raw']: #, 'mjo','rossby','kelvin','gravity']:
    fn_list = []
    for exp_num in range(1, 17):
        study_name = f"UNet_A_lat10_lead35_in_olr_tar_ROMI_mem{memory_last}_{dataflg}"
        db_path = f"sqlite:////pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/optuna/{study_name}.db"
        
        # Load the Optuna study
        study = optuna.load_study(study_name=study_name, storage=db_path)
        
        # Get best trial (this could be filtered or just the best overall)
        best_trial = study.best_trial
        params = best_trial.params

        # Extract needed params
        ch = params["num_filters_enc"]
        hidden1 = int(params["hidden_layer_first"])
        hidden2 = int(params["hidden_layer_second"])
        dropout = params["fc_dropout"]
        lr = params["learning_rate"]
        batch = params["batch_size"]
        opt = params["optimizer"]
        wd = params["weight_decay"]
        mom = params.get("momentum", 0.9)  # use 0.9 as default if not in trial
        ksize_h = params["kernel_size_h"]
        ksize_w = params["kernel_size_w"]

        # Construct file path
        filename = f"OLR_{lat_range}deg_lead{lead}_lr{lr}_batch{batch}_dropout{dropout}_ch_{ch}_ksize_{ksize_h}_{ksize_w}_hidden_{hidden1}_{hidden2}_opt_{opt}_mom{mom}_wd{wd}_mem{memory_last}.nc"
        fn = f"/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/predictions/hov_allleads/UNet_A/exp{exp_num}/{filename}"

        fn_list.append(fn)

    bcc, rmse = mjo.get_skill_all_leads_parallel('ROMI', fn_list=fn_list, rule='Iamp>1.0', month_list=None, datesta='2016-01-01', dateend='2021-12-31', lead_max=35, exp_list=np.arange(1, 17),
                        Fnmjo = '/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/target/romi/ROMI_NOAA_1979to2022.nc')

    bccs_raw = []
    for exp in np.arange(1, 17):
        bccs_raw.append(bcc[exp])

    bccs_raw = np.array(bccs_raw)

    # store the bccs_raw in a file

    fn = f"./plot_data/bccs_{dataflg}_lat{lat_range}_lead{lead}_mem{memory_last}.npy"
    np.save(fn, bccs_raw)

    print(f"bccs_{dataflg}_lat{lat_range}_lead{lead}_mem{memory_last}.npy saved in {fn}")

    # clean up
    del bcc
    del rmse
    del bccs_raw
    del fn_list
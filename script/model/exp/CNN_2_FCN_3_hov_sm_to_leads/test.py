import yaml 
import os 
import copy
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2] / 'src'))

from data_prepare.dataset import load_train_data, load_val_data
from models.cnnmlp import CNNMLP
from models.unet import UNet_A
from trainers.train import train_model_hpo
from utils.logger import setup_logger
logger = setup_logger()

import optuna
from math import ceil

# >>> CHANGE: ensemble seeds
SEEDS = [0, 1]   # 2-seed ensemble for stability


# ======================================================================
# load base config
# ======================================================================
with open("/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/config/base.yaml", "r") as f:
    base_config = yaml.safe_load(f)

base_config["dataset_type"] = "hov_sm"
base_config["data"]["dataflg"] = "raw"
base_config["data"]["input_path"] = ["/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/raw/olr.day.noaa.2x2.nc",]
base_config["data"]["memory_last"] = 95  
base_config["data"]["lat_range"] = 10
base_config["data"]["window_len"] = int(os.environ.get('window_len', 11))
base_config["data"]["residual"] = os.environ.get('residual', 'false').lower() == 'true'

base_config["model"]["cnn"]["layer_num"] = 2
base_config["model"]["mlp"]["layer_num"] = 2

with open(f'./yaml/hpo_sm{base_config["data"]["window_len"]}_res{str(base_config["data"]["residual"])}.yaml', 'w') as f:
    yaml.dump(base_config, f)

# ======================================================================
# name for optuna
# ======================================================================
dataflg = base_config["data"]["dataflg"]
model_name = base_config["model"]["name"]
lat_range = base_config["data"]["lat_range"]
lead = base_config["data"]["lead"]
memory_last = base_config["data"].get("memory_last", 0)
memory_str = str(memory_last)
input_vars = base_config["data"]["input_vars"]
target_vars = base_config["data"]["target_vars"]
input_str = "_".join(input_vars)
target_str = "_".join(target_vars)
window_len = base_config["data"]["window_len"]
residual_flg = str(base_config["data"]["residual"])

study_name = f"{model_name}_lat{lat_range}_lead{lead}_in_{input_str}_tar_{target_str}_{dataflg}_mem{memory_str}_sm{window_len}_res{residual_flg}"
optuna_dir = "./optuna"
storage_path = os.path.join(optuna_dir, f"{study_name}.db")
storage = f"sqlite:///{storage_path}"
os.makedirs(optuna_dir, exist_ok=True)


# ======================================================================
# objective
# ======================================================================
def objective(trial):
    config = copy.deepcopy(base_config)

    # ===================================================================
    # CNN parameters
    # ===================================================================
    if config["model"]["name"] == "CNN_MLP": 
        config["model"]["cnn"]["channels_list"] = [16, 16]  # no change
    elif config["model"]["name"] == "UNet_A":
        num_filters_enc = trial.suggest_categorical("num_filters_enc", [1, 2, 4])
        config["model"]["cnn"]["num_filters_enc"] = num_filters_enc

    # ===================================================================
    # kernel sizes
    # ===================================================================
    grid_res = float(config["data"]["grid_res"])
    nlon = int(np.floor(360 / grid_res))
    config["model"]["cnn"]["nlon"] = nlon

    if config["dataset_type"][:3] == "map":
        nlat = int(np.floor(lat_range / grid_res) * 2 + 1)
        config["model"]["cnn"]["nlat"] = nlat
        kernel_size_h = trial.suggest_categorical("kernel_size_h", list(range(1, nlat+1, 2)))
        kernel_size_w = trial.suggest_categorical("kernel_size_w", list(range(1, nlon+1, 4)))  # >>> CHANGE step=4 instead of 2
    elif config["dataset_type"] == "latavg":
        config["model"]["cnn"]["nlat"] = 1
        kernel_size_h = 1
        kernel_size_w = trial.suggest_categorical("kernel_size_w", list(range(1, nlon+1, 4)))  # >>> CHANGE
    else:
        kernel_size_ratio = trial.suggest_categorical("kernel_size_ratio", [8, 5])  # >>> CHANGE remove 2 (too unstable)
        nlat = min(int(memory_last+1), int(np.floor((nlon-1) / kernel_size_ratio)))
        kernel_size_h = trial.suggest_categorical("kernel_size_h", list(range(1, nlat+1, 4)))
        kernel_size_w = kernel_size_ratio * kernel_size_h + 1 

    config["model"]["cnn"]["kernel_size"] = [int(kernel_size_h), int(kernel_size_w)]
    config["kernel_size_str"] = f"{kernel_size_h}_{kernel_size_w}"

    # ===================================================================
    # MLP parameters
    # ===================================================================
    hidden_layers = [512, 256]  # fixed
    config["model"]["mlp"]["hidden_layers"] = hidden_layers
    config["hidden_layers_str"] = f"{hidden_layers[0]}_{hidden_layers[1]}"

    # ===================================================================
    # training hyperparameters (search space)
    # ===================================================================

    # >>> CHANGE: narrower LR + log-uniform
    config["training"]["learning_rate"] = trial.suggest_float(
        "learning_rate", 1e-4, 5e-3, log=True
    )

    # >>> CHANGE: dropout step
    config["model"]["mlp"]["dropout"] = trial.suggest_float(
        "fc_dropout", 0.1, 0.4, step=0.05
    )

    config["training"]["batch_size"] = trial.suggest_categorical(
        "batch_size", [16, 32, 64]
    )  # >>> CHANGE remove batch=8 (unstable gradients)

    config["training"]["optimizer"] = trial.suggest_categorical(
        "optimizer", ["SGD", "Adam"]
    )

    # >>> CHANGE: narrower weight decay
    config["training"]["weight_decay"] = trial.suggest_float(
        "weight_decay", 1e-6, 1e-3, log=True
    )

    # =====================================================================
    # Load data
    # =====================================================================
    train_loader = load_train_data(config, dataset_type=config["dataset_type"])
    val_loader = load_val_data(config, dataset_type=config["dataset_type"])

    # =====================================================================
    # Build model
    # =====================================================================
    if model_name == "CNN_MLP":
        model = CNNMLP(config["model"]["cnn"], config["model"]["mlp"])
    elif model_name == "UNet_A":
        model = UNet_A(config["model"]["cnn"], config["model"]["mlp"])
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # =====================================================================
    # ENSEMBLE LOOP
    # =====================================================================
    # >>> CHANGE: evaluate with 2 seeds and average
    scores = []
    for seed in SEEDS:
        config["training"]["seed"] = seed
        val_loss, _ = train_model_hpo(model, train_loader, val_loader, config, trial)
        scores.append(val_loss)

    # Return ensemble mean
    return float(np.mean(scores))


# =====================================================================
# Run study
# =====================================================================
if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)

    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
        )
        print(f"Loaded existing study: {study_name}")
    except KeyError:
        print(f"Creating new study: {study_name}")
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            pruner=pruner,
        )

    n_total = 200 
    n_completed = len(study.trials)
    if n_completed >= n_total:
        print(f"[INFO] Study already has {n_completed} completed trials. Nothing to do.")
        exit(0)

    n_trials_remaining = n_total - n_completed
    n_parallel = 1
    n_trials_this_worker = ceil(n_trials_remaining / n_parallel)
    print(f"[INFO] Running {n_trials_this_worker} trials on this worker.")

    if n_trials_this_worker > 0:
        study.optimize(objective, n_trials=n_trials_this_worker)
        print("[INFO] Optimization finished.")
        print("[INFO] Best trial:")
        print(study.best_trial)
    else:
        print("[INFO] No trials left to run.")

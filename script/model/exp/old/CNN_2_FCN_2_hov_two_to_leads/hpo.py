import yaml 
import os 
import copy
import numpy as np

import sys
from pathlib import Path
# Add src/ to sys.path (adjust according to your structure)
sys.path.append(str(Path(__file__).resolve().parents[2] / 'src'))

from data_prepare.dataset import load_train_data, load_val_data
from models.cnnmlp import CNNMLP
from models.unet import UNet_A
from trainers.train import train_model_hpo  # Make sure this is updated with pruning logic
from utils.logger import setup_logger
logger = setup_logger()

import optuna
from math import ceil

'''
# Example: 
# salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account=dasrepo_g
# CUDA_VISIBLE_DEVICES=1 python hpo.py
'''

# ======================================================================
# load the base configuration, modify the parameters for the experiment
# ======================================================================
with open("/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/config/base.yaml", "r") as f:
    base_config = yaml.safe_load(f)

# --- modify the base config for the experiment ---
base_config["model"]["cnn"]["layer_num"] = 2
base_config["model"]["mlp"]["layer_num"] = 1
base_config["dataset_type"] = "hov_two"
base_config["data"]["memory_last"] = 95  
base_config["model"]["cnn"]["input_channel_num"] = 2
# --- store the config under the experiment folder ---
with open('hpo.yaml', 'w') as f:
    yaml.dump(base_config, f)

# ======================================================================
# create the name for the optuna experiment
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
study_name = f"{model_name}_lat{lat_range}_lead{lead}_in_{input_str}_tar_{target_str}_{dataflg}_mem{memory_str}"
optuna_dir = "./optuna"
storage_path = os.path.join(optuna_dir, f"{study_name}.db")
storage = f"sqlite:///{storage_path}"
os.makedirs(optuna_dir, exist_ok=True)

# ======================================================================
# Optuna objective function
# ======================================================================
def objective(trial):
    config = copy.deepcopy(base_config)

    # ===================================================================
    # HPO search space
    # ===================================================================

    # ------------- hidden layers for the Convolution ---------------
    if config["model"]["name"] == "CNN_MLP": 
        channels_list = []
        for i in range(config["model"]["cnn"]["layer_num"]):
            channel = trial.suggest_categorical(f"channel_{i+1}", [4, 8, 16, 32, 64, 128])
            channels_list.append(channel)
        config["model"]["cnn"]["channels_list"] = channels_list
    elif config["model"]["name"] == "UNet_A":
        num_filters_enc = trial.suggest_categorical("num_filters_enc", [1, 2, 4, 8, 16, 32, 64])
        config["model"]["cnn"]["num_filters_enc"] = num_filters_enc

    # ------------- kernel size for the CNN ----------------
    grid_res = float(config["data"]["grid_res"])  # grid spacing in degrees to float
    nlon = int(np.floor(360 / grid_res))  # grid points in longitude
    config["model"]["cnn"]["nlon"] = nlon
    if config["dataset_type"][:3] == "map" :
        nlat = int(np.floor(lat_range / grid_res) * 2 + 1)  # grid points in latitude
        config["model"]["cnn"]["nlat"] = nlat
        kernel_size_h = trial.suggest_categorical("kernel_size_h", list(range(1, nlat+1, 2)))
        kernel_size_w = trial.suggest_categorical("kernel_size_w", list(range(1, nlon+1, 2)))
    elif config["dataset_type"] == "latavg":
        nlat = 1 
        config["model"]["cnn"]["nlat"] = nlat
        kernel_size_h = trial.suggest_categorical("kernel_size_h", [1])
        kernel_size_w = trial.suggest_categorical("kernel_size_w", list(range(1, nlon+1, 2)))
    else:
        kernel_size_ratio = 8
        nlat = min(int(memory_last+1), int(np.floor((nlon-1) / kernel_size_ratio)))
        kernel_size_h = trial.suggest_categorical("kernel_size_h", list(range(1, nlat+1, 2)))
        kernel_size_w = int(kernel_size_ratio * kernel_size_h + 1) 

    # ------------- hidden layers for the MLP ----------------
    if config["model"]["mlp"]["layer_num"] == 0:
        config["hidden_layers_str"] = "null"
        config["model"]["mlp"]["hidden_layers"] = []
    else:
        hidden_layers = []
        for i in range(config["model"]["mlp"]["layer_num"]):
            hidden_layer = trial.suggest_categorical(f"hidden_layer_{i+1}", [64, 128, 256, 512, 1024])
            hidden_layers.append(hidden_layer)
        config["model"]["mlp"]["hidden_layers"] = hidden_layers
        if len(hidden_layers) >= 2:
            config["hidden_layers_str"] = f"{hidden_layers[0]}_{hidden_layers[1]}"
        else:
            config["hidden_layers_str"] = str(hidden_layers[0])

    # ------------- training ----------------
    config["training"]["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    config["training"]["batch_size"] = trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64, 128])
    config["model"]["mlp"]["dropout"] = trial.suggest_float("fc_dropout", 0.1, 0.5)
    config["training"]["optimizer"] = trial.suggest_categorical("optimizer", ["SGD", "Adam"])
    config["training"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)


    # ====================================================================
    # define some parameters for the model
    # ====================================================================

    # ------------- kernel size for the CNN ----------------
    config["model"]["cnn"]["kernel_size"] = [int(kernel_size_h), int(kernel_size_w)]
    config["kernel_size_str"] = f"{kernel_size_h}_{kernel_size_w}"

    # ------------- input map size for the CNN ----------------
    if config["dataset_type"][:3] == "map":
        config["model"]["cnn"]["input_map_size"] = nlon * nlat
    elif config["dataset_type"] == "latavg":
        config["model"]["cnn"]["input_map_size"] = nlon 
    else:
        config["model"]["cnn"]["input_map_size"] = nlon * (1 + config["data"]["memory_last"])

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
    # train model
    # =====================================================================

    val_loss, _ = train_model_hpo(model, train_loader, val_loader, config, trial)
    return val_loss

# =====================================================================
# Run the Optuna study
# =====================================================================
if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)

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

    # Calculate remaining trials
    n_total = 200 
    # Count completed trials
    n_completed = len(study.trials)
    # Skip if no trials are left
    if n_completed >= n_total:
        print(f"[INFO] Study already has {n_completed} completed trials. Nothing to do.")
        exit(0)

    # Adjust number of trials for this worker
    n_trials_remaining = n_total - n_completed

    n_parallel = 4  # Number of parallel workers
    n_trials_this_worker = ceil(n_trials_remaining / n_parallel)
    print(f"[INFO] Running {n_trials_this_worker} trials on this worker.")

    if n_trials_this_worker > 0:
        study.optimize(objective, n_trials=n_trials_this_worker)
        print("[INFO] Optimization finished.")
        print("[INFO] Best trial:")
        print(study.best_trial)
    else:
        print("[INFO] No trials left to run.")

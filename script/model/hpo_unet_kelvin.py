import optuna 
import copy 
import yaml 
import os 
from math import ceil

from src.data_prepare.dataset import load_val_data, load_train_data
from src.models.cnnmlp import CNNMLP
from src.models.unet import UNet_A
from src.trainers.train import train_model_hpo  # Make sure this is updated with pruning logic

from src.utils.logger import setup_logger
logger = setup_logger()

# Example: salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account=dasrepo_g
# CUDA_VISIBLE_DEVICES=1 python run_optuna_hpo.py

# base configuration 
with open("/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/config/base.yaml", "r") as f:
    base_config = yaml.safe_load(f)

base_config["data"]["dataflg"] = 'kelvin'
base_config["data"]["input_path"] = '/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/processed/filtered/olr.kelvin.k1to10_T3to30.nc'

# Study naming parameters
dataflg = base_config["data"]["dataflg"]
model_name = base_config["model"]["name"]
lat_range = base_config["data"]["lat_range"]
lead = base_config["data"]["lead"]
input_vars = base_config["data"]["input_vars"]
target_vars = base_config["data"]["target_vars"]
memory_last = base_config["data"]["memory_last"]

input_str = "_".join(input_vars)
target_str = "_".join(target_vars)

study_name = f"{model_name}_lat{lat_range}_lead{lead}_in_{input_str}_tar_{target_str}_mem{memory_last}_{dataflg}"
optuna_dir = "./optuna"
storage_path = os.path.join(optuna_dir, f"{study_name}.db")
storage = f"sqlite:///{storage_path}"
os.makedirs(optuna_dir, exist_ok=True)

def objective(trial):
    config = copy.deepcopy(base_config)

    # HPO suggestions
    num_filters_enc = trial.suggest_categorical("num_filters_enc", [16, 32, 64])
    config["model"]["cnn"]["num_filters_enc"] = num_filters_enc
    hidden_layer_first = trial.suggest_categorical("hidden_layer_first", [256, 512, 1024])
    hidden_layer_second = trial.suggest_categorical("hidden_layer_second", [64, 128, 256])

    kernel_size_h = trial.suggest_categorical("kernel_size_h", [1, 3, 5, 11])
    kernel_size_w = trial.suggest_categorical("kernel_size_w", [3, 5, 9, 41, 81])
    print('warning test 50')
    config["training"]["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    config["training"]["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64])
    config["model"]["mlp"]["dropout"] = trial.suggest_float("fc_dropout", 0.1, 0.5)
    config["training"]["optimizer"] = trial.suggest_categorical("optimizer", ["SGD", "Adam"])
    config["training"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    config["model"]["mlp"]["hidden_layers"] = [int(hidden_layer_first), int(hidden_layer_second)]
    config["hidden_layers_str"] = f"{hidden_layer_first}_{hidden_layer_second}"
    config["model"]["cnn"]["kernel_size"] = [int(kernel_size_h), int(kernel_size_w)]
    config["kernel_size_str"] = f"{kernel_size_h}_{kernel_size_w}"
    config["model"]["cnn"]["input_map_size"] = config["model"]["cnn"]["nlon"] * (1 + config["data"]["memory_last"])
    print('warning test 62')
    # Load data
    train_loader = load_train_data(config, dataset_type=config["dataset_type"])
    val_loader = load_val_data(config, dataset_type=config["dataset_type"])

    # Build model
    if model_name == "CNN_MLP":
        model = CNNMLP(config["model"]["cnn"], config["model"]["mlp"])
    elif model_name == "UNet_A":
        model = UNet_A(config["model"]["cnn"], config["model"]["mlp"])
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    logger.info(f'Training with filters={num_filters_enc}, hidden={config["hidden_layers_str"]}, kernel={config["kernel_size_str"]}')
    print('warning test 76')
    val_loss, _ = train_model_hpo(model, train_loader, val_loader, config, trial)
    return val_loss

if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"Resuming existing study: {study_name}")
    except KeyError:
        study = optuna.create_study(study_name=study_name, direction="minimize", storage=storage,
                                     load_if_exists=True, pruner=pruner)
        print(f"Creating new study: {study_name}")

    # Calculate remaining trials
    n_total = 200
    n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_parallel = 3
    n_remaining = n_total - n_completed
    n_trials_per_worker = ceil(n_remaining / n_parallel)

    # Skip if no work left
    if n_trials_per_worker <= 0:
        print(f"Study already has {n_completed} complete trials. Nothing to do.")
        exit()

    # Launch just the required number per worker
    study.optimize(objective, n_trials=n_trials_per_worker)

    print("Best trial:")
    print(study.best_trial)
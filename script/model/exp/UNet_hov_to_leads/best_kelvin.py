import os 
import yaml
import copy
import optuna
import numpy as np
import sys
from math import ceil
from pathlib import Path
# Add src/ to sys.path (adjust according to your structure)
sys.path.append(str(Path(__file__).resolve().parents[2] / 'src'))
from data_prepare.dataset import load_train_data, load_val_data, load_test_data, get_time_dimension
from models.cnnmlp import CNNMLP
from models.unet import UNet_A
from trainers.train import train_model  # Make sure this is updated with pruning logic
from inference.predict import predict, predict_with_features
from utils.save_prediction import save_predictions_with_time
import torch
from utils.logger import setup_logger
logger = setup_logger()

# ======================================================================
# set the experiment number from environment variable, default to 1
# ======================================================================
# Example: salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account=dasrepo_g
# Example: salloc --nodes=4 --qos=interactive --time=04:00:00 --constraint=gpu --gpus-per-node=4 --account=dasrepo_g
exp_num = os.environ.get("exp_num", 1)

# ======================================================================
# load the base configuration 
# ======================================================================
with open("hpo_kelvin.yaml", "r") as f:
    base_config = yaml.safe_load(f)

# ======================================================================
# load the Optuna study
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
# check if the file exists
if not os.path.exists(storage_path):
    raise FileNotFoundError(f"Optuna study database not found at {storage_path}. Please run the hyperparameter optimization first.")
    exit(1)

print(f"Loading Optuna study: {study_name} from {storage_path}")
study = optuna.load_study(study_name=study_name, storage=storage)
best_params = study.best_trial.params
print(f"Best trial params:\n{best_params}")

# ======================================================================
# apply the best hyperparameters
# ======================================================================
config = copy.deepcopy(base_config)
config["exp_num"] = exp_num

# --- CNN hyperparameters ---
if model_name == "CNN_MLP":
    channels_list = []
    for i in range(config["model"]["cnn"]["layer_num"]):
        channel = best_params[f"channel_{i+1}"]
        channels_list.append(channel)
    config["model"]["cnn"]["channels_list"] = channels_list
    config["model"]["cnn"]["channels_list_str"] = "_".join(map(str, channels_list))
elif model_name == "UNet_A":
    num_filters_enc = best_params["num_filters_enc"]
    config["model"]["cnn"]["num_filters_enc"] = num_filters_enc
    config["model"]["cnn"]["channels_list_str"] = str(num_filters_enc)
# --- kernel size for the CNN ---
grid_res = float(config["data"]["grid_res"])  # grid spacing in degrees to float
nlon = int(np.floor(360 / grid_res))  # number of longitudes
config["model"]["cnn"]["nlon"] = nlon
if config["dataset_type"][:3] == "map":
    nlat = int(np.floor(lat_range / grid_res) * 2 + 1)  # number of latitudes
    config["model"]["cnn"]["nlat"] = nlat
    kernel_size_h = best_params["kernel_size_h"]
    kernel_size_w = best_params["kernel_size_w"]
elif config["dataset_type"] == "latavg":
    nlat = 1 
    config["model"]["cnn"]["nlat"] = nlat
    kernel_size_h = best_params["kernel_size_h"]
    kernel_size_w = best_params["kernel_size_w"]
else:
    kernel_size_ratio = 8 
    nlat = min(int(memory_last+1), int(np.floor((nlon-1) / kernel_size_ratio)))
    kernel_size_h = best_params["kernel_size_h"]
    kernel_size_w = int(kernel_size_ratio * kernel_size_h + 1) 
config["model"]["cnn"]["kernel_size"] = [int(kernel_size_h), int(kernel_size_w)]
config["kernel_size_str"] = f"{kernel_size_h}_{kernel_size_w}"
# --- MLP hyperparameters ---
if config["model"]["mlp"]["layer_num"] == 0:
    config["hidden_layers_str"] = "null"
    config["model"]["mlp"]["hidden_layers"] = []
else:
    hidden_layers = []
    for i in range(config["model"]["mlp"]["layer_num"]):
        hidden_layer = best_params[f"hidden_layer_{i+1}"]
        hidden_layers.append(hidden_layer)
    config["model"]["mlp"]["hidden_layers"] = hidden_layers
    if len(hidden_layers) >= 2:
        config["hidden_layers_str"] = f"{hidden_layers[0]}_{hidden_layers[1]}"
    else:
        config["hidden_layers_str"] = str(hidden_layers[0])
# --- training hyperparameters ---
config["training"]["learning_rate"] = best_params["learning_rate"]
config["training"]["batch_size"] = best_params["batch_size"]
config["training"]["optimizer"] = best_params["optimizer"]
config["training"]["weight_decay"] = best_params["weight_decay"]
config["model"]["mlp"]["dropout"] = best_params["fc_dropout"]
# --- input map size ---
if config["dataset_type"][:3] == "map":
    config["model"]["cnn"]["input_map_size"] = nlon * nlat
elif config["dataset_type"] == "latavg":
    config["model"]["cnn"]["input_map_size"] = nlon 
else:
    config["model"]["cnn"]["input_map_size"] = nlon * (1 + config["data"]["memory_last"])

# =====================================================================
# load data
# =====================================================================
train_loader = load_train_data(config, dataset_type=config["dataset_type"])
val_loader = load_val_data(config, dataset_type=config["dataset_type"])
# =====================================================================
# build model
# =====================================================================
if model_name == "CNN_MLP":
    model = CNNMLP(config["model"]["cnn"], config["model"]["mlp"])
elif model_name == "UNet_A":
    model = UNet_A(config["model"]["cnn"], config["model"]["mlp"])
else:
    raise ValueError(f"Unsupported model: {model_name}")

logger.info(f"Training best model with: {best_params}")

# =====================================================================
# train model
# =====================================================================
trained_model = train_model(model, train_loader, val_loader, config)
# =====================================================================
# save the best trained weights
# =====================================================================
config["model_save_path"] = config["model_save_path"].format(
    dataset_type=config["dataset_type"],
    exp_num=config["exp_num"],
    lat_range=config["data"]["lat_range"],
    lead=lead,
    lr=config["training"]["learning_rate"],
    batch_size=config["training"]["batch_size"],
    dropout=config["model"]["mlp"]["dropout"],
    channels_list_str=config["model"]["cnn"]["channels_list_str"],
    kernel_size=config["kernel_size_str"],
    hidden_layers_str=config["hidden_layers_str"],
    optimizer=config["training"]["optimizer"],
    momentum=config["training"]["momentum"],
    weight_decay=config["training"]["weight_decay"],
    memory_last=config["data"]["memory_last"],
    model_name=config["model"]["name"]
)
model_save_path = config["model_save_path"]
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(model.state_dict(), model_save_path)
logger.info(f"Best model saved at {model_save_path}")
# =====================================================================
# make predictions
# =====================================================================
logger.info(f"Making predictions on validation data for lead={lead}...")
test_loader = load_test_data(config, dataset_type=config["dataset_type"])
# predictions = predict(trained_model, test_loader)
trained_model = trained_model.cuda()  # Move model to GPU if available
preds, feature_maps = predict_with_features(trained_model, test_loader)

targets = torch.cat([batch[1] for batch in test_loader], dim=0).cpu()

print(feature_maps.keys())  # Layer names (e.g., 'conv1', 'encoder.1.conv', ...)

config["prediction_save_path"] = config["prediction_save_path"].format(
    dataset_type=config["dataset_type"],
    exp_num=config["exp_num"],
    lat_range=config["data"]["lat_range"],
    lead=lead,
    lr=config["training"]["learning_rate"],
    batch_size=config["training"]["batch_size"],
    dropout=config["model"]["mlp"]["dropout"],
    channels_list_str=config["model"]["cnn"]["channels_list_str"],
    kernel_size=config["kernel_size_str"],
    hidden_layers_str=config["hidden_layers_str"],
    optimizer=config["training"]["optimizer"],
    momentum=config["training"]["momentum"],
    weight_decay=config["training"]["weight_decay"],
    memory_last=config["data"]["memory_last"],
    model_name=config["model"]["name"]
)
prediction_save_path = config["prediction_save_path"]

# Get the time dimension
time = get_time_dimension(
    config["data"]["input_path"],
    config["data"]["test_start"],
    config["data"]["test_end"],
    config["data"]["lead"],
    mem=config["data"]["memory_last"]
)

# Check dimension alignment
if len(time) != preds.shape[0] or len(time) != targets.shape[0]:
    raise ValueError(f"Dimension mismatch: Time length ({len(time)}) does not match predictions ({preds.shape[0]}) or targets ({targets.shape[0]}).")

save_predictions_with_time(preds, targets, time, prediction_save_path)

logger.info(f"Predictions saved at {prediction_save_path}")

# Save feature maps
feature_maps_save_path = prediction_save_path.replace(".nc", "_feature_maps.pt")
torch.save(feature_maps, feature_maps_save_path)
logger.info(f"Feature maps saved at {feature_maps_save_path}")

# save the config file
with open(f'best_config_{config["data"]["dataflg"]}.yaml', 'w') as f:
    yaml.dump(config, f)

# =====================================================================
# cleanup
# =====================================================================
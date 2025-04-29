import optuna
import copy
import yaml
import os
import torch

from src.data_prepare.dataset import load_val_data, load_train_data, load_test_data, load_config, get_time_dimension
from src.inference.predict import predict
from src.utils.save_prediction import save_predictions_with_time
from src.models.cnnmlp import CNNMLP
from src.models.unet import UNet_A
from src.trainers.train import train_model  # plain version (no pruning)
from src.utils.logger import setup_logger

# Example: salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account=dasrepo_g
# Example: salloc --nodes=4 --qos=interactive --time=04:00:00 --constraint=gpu --gpus-per-node=4 --account=dasrepo_g

exp_num = os.environ.get("exp_num", 1)

logger = setup_logger()

# --- Load base config ---
with open("/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/config/base.yaml", "r") as f:
    base_config = yaml.safe_load(f)

# --- Load Optuna study ---
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
storage = f"sqlite:///./optuna/{study_name}.db"
study = optuna.load_study(study_name=study_name, storage=storage)

best_params = study.best_trial.params
print(f"Best trial params:\n{best_params}")

# --- Apply best hyperparameters ---
config = copy.deepcopy(base_config)
config["exp_num"] = exp_num
config["model"]["cnn"]["num_filters_enc"] = best_params["num_filters_enc"]
config["model"]["mlp"]["hidden_layers"] = [
    int(best_params["hidden_layer_first"]),
    int(best_params["hidden_layer_second"])
]
config["model"]["mlp"]["dropout"] = best_params["fc_dropout"]
config["training"]["learning_rate"] = best_params["learning_rate"]
config["training"]["batch_size"] = best_params["batch_size"]
config["training"]["optimizer"] = best_params["optimizer"]
config["training"]["weight_decay"] = best_params["weight_decay"]
config["model"]["cnn"]["kernel_size"] = [
    int(best_params["kernel_size_h"]),
    int(best_params["kernel_size_w"])
]
config["model"]["cnn"]["input_map_size"] = config["model"]["cnn"]["nlon"] * (1 + config["data"]["memory_last"])

# --- Optional: Format string versions ---
config["kernel_size_str"] = f"{best_params['kernel_size_h']}_{best_params['kernel_size_w']}"
config["hidden_layers_str"] = f"{best_params['hidden_layer_first']}_{best_params['hidden_layer_second']}"

# --- Load data ---
train_loader = load_train_data(config, dataset_type=config["dataset_type"])
val_loader = load_val_data(config, dataset_type=config["dataset_type"])

# --- Build model ---
if model_name == "CNN_MLP":
    model = CNNMLP(config["model"]["cnn"], config["model"]["mlp"])
elif model_name == "UNet_A":
    model = UNet_A(config["model"]["cnn"], config["model"]["mlp"])
else:
    raise ValueError(f"Unsupported model: {model_name}")

logger.info(f"Retraining best model with: {best_params}")

# --- Train model ---
trained_model = train_model(model, train_loader, val_loader, config)

# --- Save the trained model ---
config["model_save_path"] = config["model_save_path"].format(
    dataset_type=config["dataset_type"],
    exp_num=config["exp_num"],
    lat_range=config["data"]["lat_range"],
    lead=lead,
    lr=config["training"]["learning_rate"],
    batch_size=config["training"]["batch_size"],
    dropout=config["model"]["mlp"]["dropout"],
    channels_list_str=str(config["model"]["cnn"]["num_filters_enc"]),
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

# --- Make predictions ---
logger.info(f"Making predictions on validation data for lead={lead}...")
test_loader = load_test_data(config, dataset_type=config["dataset_type"])
predictions = predict(trained_model, test_loader)

targets = torch.cat([batch[1] for batch in test_loader], dim=0).cpu()
preds = torch.cat(predictions, dim=0).cpu()

config["prediction_save_path"] = config["prediction_save_path"].format(
    dataset_type=config["dataset_type"],
    exp_num=config["exp_num"],
    lat_range=config["data"]["lat_range"],
    lead=lead,
    lr=config["training"]["learning_rate"],
    batch_size=config["training"]["batch_size"],
    dropout=config["model"]["mlp"]["dropout"],
    channels_list_str=str(config["model"]["cnn"]["num_filters_enc"]),
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

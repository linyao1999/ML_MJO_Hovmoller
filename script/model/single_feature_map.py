import torch
import os
from src.data_prepare.dataset import MapDataset, load_config, load_val_data, get_time_dimension
from src.models.cnnmlp import CNNMLP
from src.inference.predict import predict
from src.utils.logger import setup_logger
# from src.utils.metrics import compute_metrics  # Placeholder for custom metrics, if applicable
from src.utils.save_prediction import save_predictions_with_time
import torch.nn as nn 
import argparse
# salloc --nodes 1 --qos interactive --time 02:00:00 --constraint gpu --gpus 4 --account=m4736_g

# Setup logger
logger = setup_logger()

def main(lead):
    # Load configuration
    config_path = "config/config_hovmoller.yaml"    
    # config_path = "config/config_hovmoller_kelvin.yaml"
    # config_path = "config/config_hovmoller_rossby.yaml" 
    config = load_config(config_path)

    # Update lead in the config
    config["data"]["lead"] = lead
    config["model"]["cnn"]["input_map_size"] = config["model"]["cnn"]["nlon"] * (1 + config["data"]["memory_list"][-1])

    # Dynamically substitute placeholders in `model_save_path`
    lat_range = config["data"]["lat_range"]
    lead = config["data"]["lead"]
    lr = config["training"]["learning_rate"]
    batch_size = config["training"]["batch_size"]
    memory_last = config["data"]["memory_list"][-1]

    # Format the path
    if config["pretrained_path"] is not None:
        config["pretrained_path"] = config["pretrained_path"].format(
            lat_range=lat_range,
            lead=lead,
            lr=lr,
            batch_size=batch_size,
            memory_last=memory_last
        )
    config["model_save_path"] = config["model_save_path"].format(
        lat_range=lat_range,
        lead=lead,
        lr=lr,
        batch_size=batch_size,
        memory_last=memory_last
    )
    config["prediction_save_path"] = config["prediction_save_path"].format(
        lat_range=lat_range,
        lead=lead,
        lr=lr,
        batch_size=batch_size,
        memory_last=memory_last
    )

    print(f"Model save path: {config['model_save_path']}")

    # Load the data
    val_loader = load_val_data(config, dataset_type='hov')

    # Initialize model
    cnn_config = config["model"]["cnn"]
    mlp_config = config["model"]["mlp"]
    model = CNNMLP(cnn_config, mlp_config)

    # Load the trained model
    model_path = config["model_save_path"]
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()

    # dictionary to store feature maps
    feature_maps = {}

    # hook for feature maps
    def hook_fn(module, input, output):
        feature_maps[module] = output


    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):  # Change this if you need other layers
            hooks.append(layer.register_forward_hook(hook_fn))

    model.eval()

    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            _ = model(inputs.cuda())

    # Unregister the hooks
    for hook in hooks:
        hook.remove()

    # Rename feature map keys with 'hid1', 'hid2', ...
    renamed_feature_maps = {}
    for idx, (key, value) in enumerate(feature_maps.items(), start=1):
        renamed_feature_maps[f"hid{idx}"] = value
        
    # Save the feature maps into one file
    save_path = '/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/feature_maps'
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f"feature_maps_lead_{lead}.pt")
    torch.save(renamed_feature_maps, save_path)
    print(f"Feature maps saved at {save_path}")

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate CNN-MLP model with dynamic lead.")
    parser.add_argument("--lead", type=int, required=True, help="Lead time to set in the configuration.")
    args = parser.parse_args()

    main(args.lead)

import torch
import os
import numpy as np
import xarray as xr
from src.data_prepare.dataset import load_config, load_val_data
from src.models.cnnmlp import CNNMLP
from src.utils.logger import setup_logger
import torch.nn as nn
import argparse

# best model: OLR_15deg_lead25_lr0.005_batch32_dropout0.3_ch_32_8_ksize_25_hidden_1024_128_opt_SGD_mom0.9_wd0.005_mem29.nc
# python single_feature_map.py --lead 25 --lr 0.005 --batch_size 32 --dropout 0.3 --epochs 20 --optimizer SGD --momentum 0.9 --weight_decay 0.005 --exp_num 1 --lat_range 15 --memory_last 29 --kernel_size 25
# second model: OLR_10deg_lead25_lr0.005_batch64_dropout0.3_ch_32_8_ksize_3_hidden_1024_128_opt_SGD_mom0.9_wd0.005_mem95.nc'
# python single_feature_map.py --lead 25 --lr 0.005 --batch_size 64 --dropout 0.3 --epochs 20 --optimizer SGD --momentum 0.9 --weight_decay 0.005 --exp_num 1 --lat_range 10 --memory_last 95 --kernel_size 3
# Example: salloc --nodes 1 --qos interactive --time 02:00:00 --constraint gpu --gpus 4 --account=m3312_g

# Setup logger
logger = setup_logger()

def main(
    lead,
    lr=None,
    batch_size=None,
    dropout=None,
    epochs=None,
    optimizer=None,
    momentum=None,
    weight_decay=None,
    exp_num=None,
    lat_range=None,
    memory_last=None,
    kernel_size=None,
):
    # Load configuration
    config_path = "config/config_hovmoller.yaml"
    config = load_config(config_path)
    
    # Update configuration with provided parameters
    config["data"]["lead"] = lead
    config["training"]["learning_rate"] = lr or config["training"]["learning_rate"]
    config["training"]["batch_size"] = batch_size or config["training"]["batch_size"]
    config["model"]["mlp"]["dropout"] = dropout or config["model"]["mlp"]["dropout"]
    config["training"]["epochs"] = epochs or config["training"]["epochs"]
    config["training"]["optimizer"] = optimizer or config["training"]["optimizer"]
    config["training"]["momentum"] = momentum or config["training"]["momentum"]
    config["training"]["weight_decay"] = weight_decay or config["training"]["weight_decay"]
    config["exp_num"] = exp_num or config["exp_num"]
    config["data"]["lat_range"] = lat_range or config["data"]["lat_range"]
    config["data"]["memory_last"] = memory_last or config["data"]["memory_last"]
    config["model"]["cnn"]["kernel_size"] = kernel_size or config["model"]["cnn"]["kernel_size"]

    config["model"]["cnn"]["input_map_size"] = config["model"]["cnn"]["nlon"] * (1 + config["data"]["memory_last"])
    config['channels_list_str'] = '_'.join(map(str, config["model"]["cnn"]['channels_list']))
    config['hidden_layers_str'] = '_'.join(map(str, config["model"]["mlp"]['hidden_layers']))

    config["model_save_path"] = config["model_save_path"].format(
        epochs=config["training"]["epochs"],
        exp_num=config["exp_num"],
        lat_range=config["data"]["lat_range"],
        lead=lead,
        lr=config["training"]["learning_rate"],
        batch_size=config["training"]["batch_size"],
        dropout=config["model"]["mlp"]["dropout"],
        channels_list_str=config["channels_list_str"],
        kernel_size=config["model"]["cnn"]["kernel_size"],
        hidden_layers_str=config["hidden_layers_str"],
        optimizer=config["training"]["optimizer"],
        momentum=config["training"]["momentum"],
        weight_decay=config["training"]["weight_decay"],
        memory_last=config["data"]["memory_last"]
    )

    print(f"Configuration: {config}")

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

    # Dictionary to store feature maps for all times
    feature_maps_all_times = []

    # Hook function to save feature maps
    def hook_fn(module, input, output):
        feature_maps_all_times[-1].append(output.clone().cpu().numpy())  # Save to CPU and as numpy array

    # Register hooks for all convolutional layers
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(hook_fn))

    # Evaluate model and store feature maps
    model.eval()
    with torch.no_grad():
        for inputs, _ in val_loader:
            feature_maps_all_times.append([])  # Initialize list for current batch
            _ = model(inputs.cuda())

    # Unregister the hooks
    for hook in hooks:
        hook.remove()

    # Concatenate feature maps along time dimension
    feature_maps_all_times = [
        np.concatenate(feature_maps, axis=0) for feature_maps in zip(*feature_maps_all_times)
    ]  # List of arrays: [layer1_feature_maps, layer2_feature_maps, ...]

    # Save feature maps for each layer as variables in NetCDF
    save_path = '/pscratch/sd/l/linyaoly/MJO_ML_2025/script/model/data/feature_maps'
    os.makedirs(save_path, exist_ok=True)
    file_name = os.path.basename(config["model_save_path"]).replace('.pth', '.nc')
    save_path = os.path.join(save_path, f"{file_name}")

    # Create an xarray dataset
    ds = xr.Dataset()
    
    for idx, feature_map in enumerate(feature_maps_all_times, start=1):
        print(f"Saving feature maps for layer {idx} with shape {feature_map.shape}")
        layer_name = f"hid{idx}"
        dims = ["time", f"layer{idx}", "memory", "lon"]
        
        # Add the feature map to the dataset
        ds[layer_name] = (dims[:feature_map.ndim], feature_map)
        ds[layer_name].attrs["description"] = f"Feature maps from layer {idx}"

    # Save to NetCDF
    ds.to_netcdf(save_path)
    print(f"Feature maps saved to {save_path} as NetCDF.")


# best model: OLR_15deg_lead25_lr0.005_batch32_dropout0.3_ch_32_8_ksize_25_hidden_1024_128_opt_SGD_mom0.9_wd0.005_mem29.nc
# python single_feature_map.py --lead 25 --lr 0.005 --batch_size 32 --dropout 0.3 --epochs 20 --optimizer SGD --momentum 0.9 --weight_decay 0.005 --exp_num 1 --lat_range 15 --memory_last 29 --kernel_size 25

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save feature maps from CNN-MLP model.")
    parser.add_argument("--lead", type=int, required=True, help="Lead time to set in the configuration.")
    parser.add_argument("--lr", type=float, required=False, help="Learning rate for optimization.")
    parser.add_argument("--batch_size", type=int, required=False, help="Batch size for training.")
    parser.add_argument("--dropout", type=float, required=False, help="Dropout rate for CNN.")
    parser.add_argument("--epochs", type=int, required=False, help="Number of training epochs.")
    parser.add_argument("--optimizer", type=str, required=False, choices=["SGD", "Adam"], help="Optimizer type.")
    parser.add_argument("--momentum", type=float, required=False, help="Momentum for SGD optimizer.")
    parser.add_argument("--weight_decay", type=float, required=False, help="Weight decay for regularization.")
    parser.add_argument("--exp_num", type=int, help="experiment number")
    parser.add_argument("--lat_range", type=int, help="latitude range")
    parser.add_argument("--memory_last", type=int, help="Time steps to consider in the past")
    parser.add_argument("--kernel_size", type=int, help="Kernel size for CNN.")

    args = parser.parse_args()

    main(
        lead=args.lead,
        lr=args.lr,
        batch_size=args.batch_size,
        dropout=args.dropout,
        epochs=args.epochs,
        optimizer=args.optimizer,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        exp_num=args.exp_num,
        lat_range=args.lat_range,
        memory_last=args.memory_last,
        kernel_size=args.kernel_size
    )

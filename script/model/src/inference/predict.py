import torch

def predict(model, data_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = batch
            outputs = model(inputs.cuda())
            predictions.append(outputs)
    return predictions

import torch.nn as nn

def register_conv_hooks(model, feature_maps_dict):
    """
    Registers forward hooks to all Conv2d layers in the model to store their outputs.
    Populates feature_maps_dict with layer names as keys.
    Returns a list of hook handles (so you can remove them later).
    """
    handles = []

    def hook_fn(name):
        def fn(module, input, output):
            feature_maps_dict[name] = output.detach().cpu()  # Save as CPU tensor to avoid GPU memory issues
        return fn

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(hook_fn(name)))

    return handles

from collections import defaultdict

def predict_with_features(model, data_loader, device="cuda"):
    model.eval()
    predictions = []
    # For each layer, accumulate a list of [batch, C, H, W] tensors
    feature_map_batches = defaultdict(list)

    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            feature_maps = {}   # this will hold this batch's feature maps
            handles = register_conv_hooks(model, feature_maps)
            outputs = model(inputs)
            predictions.append(outputs.cpu())
            # For each layer, append this batch's output to our accumulator
            for lname, fmap in feature_maps.items():
                feature_map_batches[lname].append(fmap)
            # Remove hooks after each batch
            for h in handles:
                h.remove()
    # Stack along time/batch dimension for each layer
    feature_maps = {k: torch.cat(v, dim=0) for k, v in feature_map_batches.items()}
    predictions = torch.cat(predictions, dim=0)
    return predictions, feature_maps

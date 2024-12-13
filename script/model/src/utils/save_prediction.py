import xarray as xr
import os

def save_predictions_with_time(predictions, targets, times, save_path):
    """
    Save predictions and targets with their corresponding time in a NetCDF file.
    
    Args:
        predictions (torch.Tensor): Model predictions (N, ...).
        targets (torch.Tensor): Ground truth values (N, ...).
        times (array-like): Corresponding timestamps (N,).
        save_path (str): Path to save the NetCDF file.
    """
    # Convert predictions and targets to xarray DataArrays
    preds_da = xr.DataArray(predictions.numpy(), dims=["time", "variable"], coords={"time": times})
    targets_da = xr.DataArray(targets.numpy(), dims=["time", "variable"], coords={"time": times})

    # Create an xarray Dataset
    dataset = xr.Dataset({
        "predictions": preds_da,
        "targets": targets_da
    })

    # Save to NetCDF
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dataset.to_netcdf(save_path)
    print(f"Predictions and targets saved to {save_path}")

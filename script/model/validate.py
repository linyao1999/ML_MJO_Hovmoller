import torch
import os
from src.data_prepare.dataset import MapDataset, load_config, load_val_data, get_time_dimension
from src.models.cnnmlp import CNNMLP
from src.inference.predict import predict
from src.utils.logger import setup_logger
# from src.utils.metrics import compute_metrics  # Placeholder for custom metrics, if applicable
from src.utils.save_prediction import save_predictions_with_time

# salloc --nodes 1 --qos interactive --time 02:00:00 --constraint gpu --gpus 4 --account=m4736_g

# Setup logger
logger = setup_logger()

def main():
    # Load configuration
    config_path = "config/config_hovmoller.yaml"
    config = load_config(config_path)

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
    # Make predictions
    logger.info("Making predictions on validation data...")
    predictions = predict(model, val_loader)

    # Evaluate predictions
    logger.info("Evaluating predictions...")
    # Concatenate targets and predictions
    targets = torch.cat([batch[1] for batch in val_loader], dim=0).cpu().squeeze()
    preds = torch.cat(predictions, dim=0).cpu()

    print(f"Predictions shape: {preds.shape}, Targets shape: {targets.shape}")
    # Save predictions and targets
    save_prediction_path = config["prediction_save_path"]

    # Get the time dimension
    time = get_time_dimension(
        config["data"]["input_path"],
        config["data"]["val_start"],
        config["data"]["val_end"],
        config["data"]["lead"],
        mem=config["data"]["memory_list"][-1]
    )

    # Check dimension alignment
    if len(time) != preds.shape[0] or len(time) != targets.shape[0]:
        raise ValueError(f"Dimension mismatch: Time length ({len(time)}) does not match predictions ({preds.shape[0]}) or targets ({targets.shape[0]}).")

    save_predictions_with_time(preds, targets, time, save_prediction_path)

    logger.info(f"Predictions saved at {save_prediction_path}")



if __name__ == "__main__":
    main()

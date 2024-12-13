import torch
import os
from src.data_prepare.dataset import MapDataset, load_config, load_val_data, load_train_data, get_time_dimension
from src.models.cnnmlp import CNNMLP
from src.trainers.train import train_model
from src.inference.predict import predict
from src.utils.logger import setup_logger
# from src.utils.metrics import compute_metrics  # Placeholder for custom metrics, if applicable
from src.utils.save_prediction import save_predictions_with_time

# salloc --nodes 1 --qos interactive --time 02:00:00 --constraint gpu --gpus 4 --account=m4736_g


# Setup logger
logger = setup_logger()

def main():
    # Load configuration
    config_path = "config/config_map.yaml"
    config = load_config(config_path)

    # Load the data
    train_loader = load_train_data(config)
    val_loader = load_val_data(config)

    # Initialize model
    cnn_config = config["model"]["cnn"]
    mlp_config = config["model"]["mlp"]
    model = CNNMLP(cnn_config, mlp_config)

    # Train the model
    logger.info("Starting training...")
    trained_model = train_model(model, train_loader, val_loader, config)

    # Save the model
    model_save_path = config["model_save_path"]
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(trained_model.state_dict(), model_save_path)
    logger.info(f"Best model saved at {model_save_path}")

    # Make predictions
    logger.info("Making predictions on validation data...")
    predictions = predict(trained_model, val_loader)

    # Evaluate predictions
    logger.info("Evaluating predictions...")
    # Concatenate targets and predictions
    targets = torch.cat([batch[1] for batch in val_loader], dim=0).cpu()
    preds = torch.cat(predictions, dim=0).cpu()

    # Save predictions and targets
    save_prediction_path = config["prediction_save_path"]

    # Get the time dimension
    time = get_time_dimension(
        config["data"]["input_path"],
        config["data"]["val_start"],
        config["data"]["val_end"],
        config["data"]["lead"]
    )

    # Check dimension alignment
    if len(time) != preds.shape[0] or len(time) != targets.shape[0]:
        raise ValueError(f"Dimension mismatch: Time length ({len(time)}) does not match predictions ({preds.shape[0]}) or targets ({targets.shape[0]}).")

    save_predictions_with_time(preds, targets, time, save_prediction_path)

    logger.info(f"Predictions saved at {save_prediction_path}")



if __name__ == "__main__":
    main()

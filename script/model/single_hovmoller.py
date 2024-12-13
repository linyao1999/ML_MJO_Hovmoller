import torch
import os
import argparse
from src.data_prepare.dataset import load_config, load_val_data, load_train_data, get_time_dimension
from src.models.cnnmlp import CNNMLP
from src.trainers.train import train_model
from src.inference.predict import predict
from src.utils.logger import setup_logger
from src.utils.save_prediction import save_predictions_with_time

# Setup logger
logger = setup_logger()

def main(lead):
    # Load configuration
    config_path = "config/config_hovmoller.yaml"
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
    train_loader = load_train_data(config, dataset_type='hov')
    val_loader = load_val_data(config, dataset_type='hov')

    # Initialize model
    cnn_config = config["model"]["cnn"]
    mlp_config = config["model"]["mlp"]
    model = CNNMLP(cnn_config, mlp_config)

    # Train the model
    logger.info(f"Starting training for lead={lead}...")
    trained_model = train_model(model, train_loader, val_loader, config)

    # Save the model
    model_save_path = config["model_save_path"]
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(trained_model.state_dict(), model_save_path)
    logger.info(f"Best model saved at {model_save_path}")

    # Make predictions
    logger.info(f"Making predictions on validation data for lead={lead}...")
    predictions = predict(trained_model, val_loader)

    # Evaluate predictions
    logger.info(f"Evaluating predictions for lead={lead}...")
    targets = torch.cat([batch[1] for batch in val_loader], dim=0).cpu()
    preds = torch.cat(predictions, dim=0).cpu()

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
    parser = argparse.ArgumentParser(description="Train and evaluate CNN-MLP model with dynamic lead.")
    parser.add_argument("--lead", type=int, required=True, help="Lead time to set in the configuration.")
    args = parser.parse_args()

    main(args.lead)

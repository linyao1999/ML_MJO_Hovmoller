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

    config["prediction_save_path"] = config["prediction_save_path"].format(
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

    if config["pretrained_path"] is not None:
        config["pretrained_path"] = config["pretrained_path"].format(
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
        mem=config["data"]["memory_last"]
    )

    # Check dimension alignment
    if len(time) != preds.shape[0] or len(time) != targets.shape[0]:
        raise ValueError(f"Dimension mismatch: Time length ({len(time)}) does not match predictions ({preds.shape[0]}) or targets ({targets.shape[0]}).")

    save_predictions_with_time(preds, targets, time, save_prediction_path)

    logger.info(f"Predictions saved at {save_prediction_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate CNN-MLP model with dynamic lead.")
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



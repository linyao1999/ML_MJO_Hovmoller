import torch
import torch.nn as nn
import torch.optim as optim
import math
import optuna   

def train_model(model, train_loader, val_loader, config):

    # NEW: Check if a pretrained model path is provided
    pretrained_path = config.get("pretrained_path", None)
    if pretrained_path:
        print(f"Loading model parameters from: {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path))
    else:
        print("Training model from scratch.")

    # Move model to GPU
    model = model.cuda()

    # Default values for training configuration
    default_config = {
        "learning_rate": 0.001,
        "epochs": 20,
        "optimizer": "Adam",  # Default optimizer
        "criterion": "MSELoss",  # Default loss function
        "scheduler": {
            "type": "CosineAnnealingLR",  # Default scheduler
        }
    }

    # Merge default config with user-provided config
    training_config = {**default_config, **config.get("training", {})}

    # Initialize criterion (loss function)
    criterion = getattr(nn, training_config["criterion"])()

    # Initialize optimizer
    if training_config["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=training_config["learning_rate"], weight_decay=training_config.get("weight_decay", 1e-4))
    elif training_config["optimizer"] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=training_config["learning_rate"], momentum=0.9, weight_decay=training_config.get("weight_decay", 1e-4))
    
    # Initialize scheduler (if provided in the config)
    scheduler = None
    if "scheduler" in training_config and training_config["scheduler"]:
        scheduler_config = training_config["scheduler"]
        scheduler_type = getattr(optim.lr_scheduler, scheduler_config["type"])
        
        # Add 'T_max' to the scheduler configuration if using CosineAnnealingLR
        if scheduler_config["type"] == "CosineAnnealingLR":
            scheduler_config["T_max"] = training_config["epochs"]
        
        scheduler = scheduler_type(optimizer, **{key: value for key, value in scheduler_config.items() if key != "type"})

    # Count trainable parameters in the model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters in the model: {num_params}")

    # Variables for tracking the best validation loss
    best_val_loss = float("inf")
    best_model_state = None

    patience = training_config.get("early_stopping_patience", None)
    no_improve_epochs = 0

    # Training loop
    for epoch in range(training_config["epochs"]):
        print(f"Epoch {epoch+1}/{training_config['epochs']}")
        model.train()  # Set model to training mode
        running_loss = 0.0

        for batch in train_loader:
            inputs, targets = batch
            # print(f"Input shape from DataLoader: {inputs.shape}")  # Expect [batch_size, channels, height, width]
    
            inputs, targets = inputs.cuda(), targets.cuda()  # Move data to GPU

            optimizer.zero_grad()  # Zero gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs.squeeze(), targets.squeeze())  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters

            running_loss += loss.item()  # Accumulate loss

            # print(f"Batch Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_loader)

        # Validation loop
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # Disable gradient calculations for validation
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.cuda(), targets.cuda()  # Move data to GPU

                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets.squeeze())

                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Print epoch results
        print(f"Epoch {epoch+1}/{training_config['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Check for NaN or infinite validation loss
        if math.isnan(val_loss):
            print(f"Epoch {epoch+1}: Validation loss is NaN. Stopping training.")
            break
        if math.isinf(val_loss):
            print(f"Epoch {epoch+1}: Validation loss is infinite. Stopping training.")
            break

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # Update learning rate scheduler
        if scheduler:
            scheduler.step()

        if patience and no_improve_epochs >= patience:
            print(f"Early stopping triggered after {no_improve_epochs} epochs without improvement.")
            break

    # Load the best model state (optional, based on use case)
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model

def train_model_hpo(model, train_loader, val_loader, config, trial):

    model = model.cuda()

    default_config = {
        "learning_rate": 0.001,
        "epochs": 20,
        "optimizer": "Adam",
        "criterion": "MSELoss",
        "scheduler": {
            "type": "CosineAnnealingLR",
        }
    }

    training_config = {**default_config, **config.get("training", {})}
    criterion = getattr(nn, training_config["criterion"])()

    if training_config["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=training_config["learning_rate"],
                               weight_decay=training_config.get("weight_decay", 1e-4))
    elif training_config["optimizer"] == "SGD":
        optimizer = optim.SGD(model.parameters(),
                              lr=training_config["learning_rate"],
                              momentum=training_config.get("momentum", 0.9),
                              weight_decay=training_config.get("weight_decay", 1e-4))

    scheduler = None
    if "scheduler" in training_config and training_config["scheduler"]:
        scheduler_config = training_config["scheduler"]
        scheduler_type = getattr(optim.lr_scheduler, scheduler_config["type"])
        if scheduler_config["type"] == "CosineAnnealingLR":
            scheduler_config["T_max"] = training_config["epochs"]
        scheduler = scheduler_type(optimizer, **{k: v for k, v in scheduler_config.items() if k != "type"})

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters in the model: {num_params}")

    best_val_loss = float("inf")
    best_model_state = None
    patience = training_config.get("early_stopping_patience", None)
    no_improve_epochs = 0
    val_loss_history = []

    for epoch in range(training_config["epochs"]):
        print(f"Epoch {epoch+1}/{training_config['epochs']}")
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets.squeeze())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets.squeeze())
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_loss_history.append(val_loss)

        print(f"Epoch {epoch+1}/{training_config['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Prune if the validation loss is NaN
        if math.isnan(val_loss):
            print(f"Epoch {epoch+1}: Validation loss is NaN. Pruning trial.")
            trial.report(val_loss, step=epoch)
            raise optuna.TrialPruned()

        # Prune if the validation loss is infinite
        if math.isinf(val_loss):
            print(f"Epoch {epoch+1}: Validation loss is infinite. Pruning trial.")
            trial.report(val_loss, step=epoch)
            raise optuna.TrialPruned()

        trial.report(val_loss, step=epoch)
        if trial.should_prune():
            print("Trial pruned.")
            raise optuna.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if scheduler:
            scheduler.step()

        if patience and no_improve_epochs >= patience:
            print(f"Early stopping triggered after {no_improve_epochs} epochs without improvement.")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)

    return best_val_loss, val_loss_history

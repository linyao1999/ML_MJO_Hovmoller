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

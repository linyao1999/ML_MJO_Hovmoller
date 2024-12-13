import torch.nn as nn
import torch
from .cnn import CNN
from .mlp import MLP

class CNNMLP(nn.Module):
    def __init__(self, cnn_config, mlp_config):
        """
        Parameters:
        - cnn_config: Dictionary with CNN parameters.
          Example:
          {
              "input_channel_num": 3,
              "channels_list": [16, 32, 64],
              "kernel_size": 3,
              "stride": 1,
              "padding": 1,
              "dropout": 0.1,
              "batch_norm": True
          }
        - mlp_config: Dictionary with MLP parameters.
          Example:
          {
              "input_size": Flattened size after CNN,
              "hidden_layers": [128, 64],
              "output_size": 10,
              "dropout": 0.5
          }
        """
        super(CNNMLP, self).__init__()

        # Initialize the CNN
        self.cnn = CNN(
            input_channel_num=cnn_config["input_channel_num"],
            channels_list=cnn_config["channels_list"],
            kernel_size=cnn_config.get("kernel_size", 5),
            stride=cnn_config.get("stride", 1),
            padding=cnn_config.get("padding", 'same'),
            dropout=cnn_config.get("dropout", 0.2),
            batch_norm=cnn_config.get("batch_norm", True)
        )

        # Compute the flattened size after CNN
        # Dummy input to calculate the shape
        flattened_size = cnn_config['channels_list'][-1] * cnn_config['input_map_size']

        # Initialize the MLP
        self.mlp = MLP(
            input_size=flattened_size,
            hidden_layers=mlp_config["hidden_layers"],
            output_size=mlp_config["output_size"],
            dropout=mlp_config.get("dropout", 0.5)
        )

    def forward(self, x):
        x = self.cnn(x)  # Pass through CNN
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.mlp(x)  # Pass through MLP
        return x

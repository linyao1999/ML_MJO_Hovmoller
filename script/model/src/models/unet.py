import torch.nn as nn
import torch
from .cnn import CNN_one
from .mlp import MLP

# CNN: multiple layer convolutional neural network, 
# each layer has a convolutional layer, batch normalization, ReLU activation function, and dropout

class UNet_A(nn.Module): # architecture from Ashesh
    def __init__(self, cnn_config, mlp_config):
        """
        Parameters:
        - cnn_config: Dictionary with CNN parameters.
          Example:
          {
              "input_channel_num": 3,
              "kernel_size": [3,3] or [5,40]
              "stride": 1,
              "padding": 1,
              "dropout": 0.1,
              "batch_norm": True
          }
        - mlp_config: Dictionary with MLP parameters.
          Example:
          {
              "input_size": Flattened size after CNN,
              "output_size": 10,
              "dropout": 0.5
          }
        """
        super(UNet_A, self).__init__()

        num_filters_enc = cnn_config.get("num_filters_enc", 64)
        num_filters_dec1 = int(num_filters_enc * 2)
        num_filters_dec2 = int(num_filters_enc * 3)
        # Initialize the CNN
        kernel_size_list = cnn_config.get("kernel_size", [5,5])
        print('kernel_size_list:', kernel_size_list)
        print('num_filters_enc:', num_filters_enc)
        print('num_filters_dec1:', num_filters_dec1)
        print('num_filters_dec2:', num_filters_dec2)

        self.hid1 = CNN_one(
            input_channel_num=cnn_config["input_channel_num"],
            output_channel_num=num_filters_enc,
            kernel_size=(kernel_size_list[0], kernel_size_list[1]),
            stride=cnn_config.get("stride", 1),
            padding=cnn_config.get("padding", 'same'),
            dropout=cnn_config.get("dropout", 0.2),
            batch_norm=cnn_config.get("batch_norm", True)
        )

        self.hid2 = CNN_one(
            input_channel_num=num_filters_enc,
            output_channel_num=num_filters_enc,
            kernel_size=(kernel_size_list[0], kernel_size_list[1]),
            stride=cnn_config.get("stride", 1),
            padding=cnn_config.get("padding", 'same'),
            dropout=cnn_config.get("dropout", 0.2),
            batch_norm=cnn_config.get("batch_norm", True)
        )

        self.hid3 = CNN_one(
            input_channel_num=num_filters_enc,
            output_channel_num=num_filters_enc,
            kernel_size=(kernel_size_list[0], kernel_size_list[1]),
            stride=cnn_config.get("stride", 1),
            padding=cnn_config.get("padding", 'same'),
            dropout=cnn_config.get("dropout", 0.2),
            batch_norm=cnn_config.get("batch_norm", True)
        )

        self.hid4 = CNN_one(
            input_channel_num=num_filters_enc,
            output_channel_num=num_filters_enc,
            kernel_size=(kernel_size_list[0], kernel_size_list[1]),
            stride=cnn_config.get("stride", 1),
            padding=cnn_config.get("padding", 'same'),
            dropout=cnn_config.get("dropout", 0.2),
            batch_norm=cnn_config.get("batch_norm", True)
        )

        self.hid5 = CNN_one(
            input_channel_num=num_filters_enc,
            output_channel_num=num_filters_enc,
            kernel_size=(kernel_size_list[0], kernel_size_list[1]),
            stride=cnn_config.get("stride", 1),
            padding=cnn_config.get("padding", 'same'),
            dropout=cnn_config.get("dropout", 0.2),
            batch_norm=cnn_config.get("batch_norm", True)
        )

        self.hid6 = CNN_one(
            input_channel_num=num_filters_dec1,
            output_channel_num=num_filters_dec1,
            kernel_size=(kernel_size_list[0], kernel_size_list[1]),
            stride=cnn_config.get("stride", 1),
            padding=cnn_config.get("padding", 'same'),
            dropout=cnn_config.get("dropout", 0.2),
            batch_norm=cnn_config.get("batch_norm", True)
        )

        # flatten_size = num_filters_dec2 * cnn_config['input_map_size']
        flatten_size = num_filters_enc * cnn_config['input_map_size']

        self.mlp = MLP(
            input_size=flatten_size,
            hidden_layers=mlp_config["hidden_layers"],
            output_size=mlp_config["output_size"],
            dropout=mlp_config.get("dropout", 0.5)
        )

        print('Initialized UNet_A')

    def forward(self, x):
        x = self.hid1(x)
        x2 = self.hid2(x)
        x3 = self.hid3(x2)
        x4 = self.hid4(x3)
        x5 = self.hid5(x4)

        # concatencate x5 and x3
        x3p5 = torch.cat((x5, x3), dim=1) # dim=1 is the channel dimension

        # print(x3p5.shape)

        x6 = self.hid6(x3p5)

        x2p6 = torch.cat((x6, x2), dim=1)

        # x = x2p6.view(x2p6.size(0), -1)
        x = x5.view(x5.size(0), -1)

        x = self.mlp(x)

        return x



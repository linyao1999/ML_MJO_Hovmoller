import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_channel_num, channels_list, kernel_size=5, 
                 stride=1, padding='same', dropout=0.2, batch_norm=True):
        """
        Parameters:
        - input_channel_num: Number of input channels (e.g., 3 for RGB images).
        - channels_list: List of output channels for each convolutional layer.
          Example: [16, 32, 64] creates 3 layers with 16, 32, and 64 output channels respectively.
        - kernel_size: Kernel size for all layers (default: 3).
        - stride: Stride for all layers (default: 1).
        - padding: Padding for all layers (default: 1).
        - dropout: Dropout rate applied after each convolutional layer (default: 0.1).
        - batch_norm: Whether to include batch normalization after each convolutional layer (default: True).
        """
        super(CNN, self).__init__()
        
        layers = []
        in_channels = input_channel_num
        
        for out_channels in channels_list:
            # Add convolutional layer
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
            )
            
            # Add batch normalization if enabled
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            
            # Add activation function (ReLU)
            layers.append(nn.ReLU())
            
            # Add dropout
            layers.append(nn.Dropout2d(dropout))
            
            # Update in_channels for the next layer
            in_channels = out_channels
        
        # Combine all layers into a sequential module
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

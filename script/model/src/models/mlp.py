import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout=0.5):
        super(MLP, self).__init__()
        layers = []
        in_features = input_size
        for hidden in hidden_layers:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = hidden
        layers.append(nn.Linear(in_features, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

        

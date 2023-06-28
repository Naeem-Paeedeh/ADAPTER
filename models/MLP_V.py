from torch import nn
from configs.configs_training import ConfigurationTraining


class MLP_V(nn.Module):
    """MLP with arbitrary number of layers"""
    def __init__(self,
                 configs: ConfigurationTraining,
                 size_first_layer,
                 size_hidden_layer,
                 size_last_layer):
        """An MLP with a  number of layers.

        Args:
            input_size (_type_): The number of activations in the first layer
            hidden_size (_type_): The number of neurons in hidden layers
            output_size (_type_): The number of
            n_layers (_type_): The number of layers.
            device (_type_): CPU or GPU
            dropout_rate (float, optional): The dropout rate. Defaults to 0.0.
        """
        super().__init__()

        self.dropout_rate = configs.dropout_rate_classifier_head
        device = configs.device
        self.n_layers = configs.n_layers_classifier

        self.dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None

        assert self.n_layers >= 1

        if self.n_layers > 1:
            self.input_layer = nn.Linear(size_first_layer, size_hidden_layer, device=device)

            self.hidden_layers = nn.ModuleList([
                nn.Linear(size_hidden_layer, size_hidden_layer, device=device) for _ in range(self.n_layers - 2)
            ])

            self.output_layer = nn.Linear(size_hidden_layer, size_last_layer, device=device)
            self.relu = nn.ReLU()
        else:
            # When we want to define a linear layer
            self.input_layer = nn.Linear(size_first_layer, size_last_layer, device=device)

    def forward(self, x):
        if self.n_layers > 1:
            x = self.input_layer(x)

            for layer in self.hidden_layers:
                if self.dropout_rate > 0.0:
                    x = self.dropout(x)
                x = self.relu(x)
                x = layer(x)

            if self.dropout_rate > 0.0:
                x = self.dropout(x)
            x = self.relu(x)
            x = self.output_layer(x)
        else:
            x = self.input_layer(x)

        return x

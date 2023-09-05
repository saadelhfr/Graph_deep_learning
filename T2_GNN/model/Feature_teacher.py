import torch
import torch.nn as nn
import torch.nn.functional as F

import os

# Setting environment variable to debug CUDA errors
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class Teacher_Features(nn.Module):
    def __init__(self, nbr_nodes: int, in_channels: int, out_channels: int,
                 hid_channels: int, dropout: float, device: torch.device) -> None:
        """
        Initialize the Teacher_Features class.
        
        Parameters:
        nbr_nodes: Number of nodes in the graph
        in_channels: Number of input features per node
        out_channels: Number of output features per node
        hid_channels: Number of hidden channels
        dropout: Dropout rate
        device: Device to which tensors should be moved (CPU or CUDA)
        """
        super(Teacher_Features, self).__init__()
        self.device = device
        self.dropout = dropout

        # Initialize the importance features as a parameter tensor
        self.imp_features = nn.Parameter(torch.empty(size=(nbr_nodes, in_channels)))
        nn.init.xavier_uniform_(self.imp_features.data, gain=1.414)

        # Define linear layers
        self.linear1 = nn.Linear(in_channels, hid_channels)
        self.linear2 = nn.Linear(hid_channels, hid_channels)
        self.linear3 = nn.Linear(hid_channels, out_channels)

        # Initialize weights
        self.weights_init()
        self.to(self.device)

    def weights_init(self) -> None:
        """Initializes the weights of linear layers."""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight.data, mode='fan_in', nonlinearity='leaky_relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias.data)

    def forward(self, x: torch.Tensor, pe_feat: torch.Tensor) :
        """
        Forward pass of the model.

        Parameters:
        x: Input features tensor
        pe_feat: Additional features (could be positional encoding, or any other kind of features)

        Returns:
        h3: Output feature tensor
        middle_representation: List of intermediate feature tensors
        """
        # Obtain indices from the sparse tensor pe_feat
        idx = pe_feat._indices()[1]

        # Filter important features according to the indices
        imp_feat_reduced = self.imp_features[idx]

        # Replace NaN values in x with corresponding values from imp_feat_reduced
        nan_mask = torch.isnan(x)
        x[nan_mask] = imp_feat_reduced[nan_mask]

        # Move the tensor to the specified device
        x.to(self.device)

        # List to store intermediate feature representations
        middle_representation = []

        # First linear layer
        h1 = self.linear1(x)
        middle_representation.append(h1)

        # Apply dropout and LeakyReLU activation
        h2 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = F.leaky_relu(self.linear2(h2))
        middle_representation.append(h2)

        # Apply another dropout and LeakyReLU activation
        h3 = F.dropout(h2, p=self.dropout, training=self.training)
        h3 = F.leaky_relu(self.linear3(h3))
        middle_representation.append(h3)

        return h3, middle_representation

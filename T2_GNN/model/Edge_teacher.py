import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv
from torch.utils.checkpoint import checkpoint
import os

# Set environment variable to debug CUDA errors
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Teacher_Edge(nn.Module):
    def __init__(self, nbr_nodes: int, in_channels: int, out_channels: int, 
                 hid_channels: int, dropout: float, device: torch.device) -> None:
        """
        Initializes the Teacher_Edge class.
        
        Parameters:
        nbr_nodes: Number of nodes in the graph
        in_channels: Number of input features per node
        out_channels: Number of output features per node
        hid_channels: Number of hidden channels
        dropout: Dropout rate
        device: Computing device (CPU or CUDA)
        """
        super(Teacher_Edge, self).__init__()
        self.nbr_nodes = nbr_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.device = device

        # Define Graph Convolutional Layers
        self.gcn1 = DenseSAGEConv(in_channels, hid_channels)
        self.gcn2 = DenseSAGEConv(hid_channels, hid_channels)
        self.gcn3 = DenseSAGEConv(hid_channels, out_channels)

        # Define a linear layer
        self.linear = nn.Linear(self.nbr_nodes, self.in_channels, bias=True)

        # Move the model to the device (CPU or GPU)
        self.to(self.device)
    
    def forward(self, Adj: torch.Tensor, pe_feat: torch.Tensor, X: torch.Tensor) :
        """
        Forward pass of the model.
        
        Parameters:
        Adj: Adjacency Matrix
        pe_feat: Input feature matrix
        X: Additional feature matrix
        
        Returns:
        h3: Output feature representation
        middle_representation: List of intermediate feature representations
        """
        middle_representation = []  # To store intermediate representations

        # Linear transformation of pe_feat
        x = self.linear(pe_feat)

        # Replace NaN values in X with corresponding values from x
        mask = torch.isnan(X)
        X[mask] = x[mask]

        # First Graph Convolutional layer
        h1 = self.gcn1(x, Adj)
        middle_representation.append(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h1 = F.leaky_relu(h1)

        # Second Graph Convolutional layer
        h2 = self.gcn2(h1, Adj)
        middle_representation.append(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        h2 = F.leaky_relu(h2)

        # Third Graph Convolutional layer
        h3 = self.gcn3(h2, Adj)
        middle_representation.append(h3)

        return h3, middle_representation

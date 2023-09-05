import torch 
import math
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv, DenseGCNConv , DenseGATConv , SAGEConv , GCNConv , GATConv
from torch_geometric.data import Data 
import random
from torch_geometric.nn.dense import diff_pool as DiffPool
'''
We define the Hyrerarchoical Graph Neural Network (HGNN) model in this file.
the modelcomports multiple blocks of a GNN and a pooling layer.
we provide the loss functon , the training , the aggregation , sampling and the evaluation functions.
we provide the model with the following parameters:
    - num_blocks : number of GNN+pooling blocks
    - num_layers: number of GNN layers
    - ratio : ratio of nodes to keep after pooling
    - input_dim: number of input features
    - otimizer: optimizer to use
    - device: device to use
    - hidden: number of hidden units
    - dropout: dropout rate
    - pooling: pooling method to use
    - GNN : GNN to use
    - loss_fn: loss function to use
    - activation: activation function to use
    - size: size of the dataset

'''

class GNBlock(nn.Module):
    def __init__(self , embGNN  ,asgGNN , device ) :
        super(GNBlock, self).__init__()
        # initialize the weight of the GNNs
        self.embGNN = embGNN
        self.asgGNN = asgGNN

        self.to(device)

    def forward(self , x , adj ) :
        # forward pass of the model
        Xemb1 , Xemb2 = self.embGNN(x , adj)
        S = self.asgGNN(Xemb1 , adj)
        return Xemb1 ,Xemb2 , S 

class embGNN(nn.Module):  # corrected 'nn.module' to 'nn.Module'
    def __init__(self, input_dim , hidden , hidden2, device , dropout_rate=0.2):
        super(embGNN, self).__init__()
        self.conv1 = DenseSAGEConv(input_dim , hidden)
        self.conv2 = DenseSAGEConv(hidden , hidden2)
        self.conv3 = DenseSAGEConv(hidden2 , input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.LeakyReLU(0.2 , inplace=True)  # corrected 'LeakyRelU' to 'LeakyReLU'
        self.to(device)
    
    def forward(self , x , adj):
        x1 = self.dropout(self.activation(self.conv1(x , adj)))
        x2 = self.dropout(self.activation(self.conv2(x1 , adj)))
        x = self.dropout(self.activation(self.conv3(x2 , adj)))
        return x , x2
    
class asgGNN(nn.Module):  # corrected 'nn.module' to 'nn.Module'
    def __init__(self, input_dim , nbr_clusters , device ):
        super(asgGNN, self).__init__()
        self.conv = DenseSAGEConv(input_dim , nbr_clusters)
        self.to(device)

    def forward(self , x , adj):
        S = self.conv(x , adj)
        S = F.softmax(S, dim=1)  # added softmax along the correct dimension
        return S

class asgGNN2(nn.Module):
    def __init__(self, input_dim , nbr_clusters , device ):
        super(asgGNN2, self).__init__()
        self.linear = nn.Linear(input_dim , nbr_clusters)
        self.to(device)

    def forward(self , x , adj):
        S = self.linear(x)
        S = F.softmax(S, dim=1)  # added softmax along the correct dimension
        return S
    

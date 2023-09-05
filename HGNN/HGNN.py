

import torch 
import math
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv, DenseGCNConv , DenseGATConv , SAGEConv , GCNConv , GATConv
from torch_geometric.data import Data 
import random
from torch_geometric.nn.dense import diff_pool as DiffPool
from .blocks import GNBlock
from .blocks import embGNN , asgGNN , asgGNN2



class HGNN(nn.Module):
    def __init__(self , n_nodes , ratio , device , input_dim , hidden_dim ): 
        super(HGNN, self).__init__()
        self.n_nodes = n_nodes
        self.ratio = ratio
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.build()

    def build(self ):
        self.embGNN = embGNN(self.input_dim , self.hidden_dim , self.hidden_dim , self.device)
        self.asgGNN = asgGNN(self.input_dim ,  math.floor(self.ratio*self.n_nodes) , self.device)
        self.asgGNN2 = asgGNN2(self.input_dim ,  math.floor(self.ratio*self.n_nodes) , self.device)
        self.a = nn.Parameter(torch.Tensor([0.5]))
        self.block = GNBlock(self.embGNN , self.asgGNN , self.device)
        self.to(self.device)

    def forward(self , data):
        x = data.x.to(self.device)
        adj = data.adj.to(self.device)
        
        x_emb1 , x_emb2 , S1 = self.block(x , adj)
        S2 = self.asgGNN2(x , adj)
        S = self.a * S1 + (1-self.a) * S2
        x_cluster1 , adj1 , link_loss1, entropy_loss1  = DiffPool.dense_diff_pool(x_emb1,adj , S1)
        x_cluster2 , adj2 ,link_loss2 , entropy_loss2   = DiffPool.dense_diff_pool(x_emb1 ,adj, S2)
        x_cluster ,  adj , link_loss, entropy_loss = DiffPool.dense_diff_pool(x_emb1 ,adj, S)
        # student's t kernel 
        dist = torch.cdist(x_emb1.squeeze(0), x_cluster, p=2)**2
        dist1 = torch.cdist(x_emb1.squeeze(0), x_cluster1, p=2)**2
        dist2 = torch.cdist(x_emb1.squeeze(0), x_cluster2, p=2)**2

        # Form Student's t-distribution-based similarity
        v = 1.0  # degrees of freedom, adjust as necessary
        Q = 1.0 / (1.0 + dist / v)
        Q = Q.pow((v + 1.0) / 2.0)
        Q = Q / torch.sum(Q, dim=1, keepdim=True)

        Q1 = 1.0 / (1.0 + dist1 / v)
        Q1 = Q1.pow((v + 1.0) / 2.0)
        Q1 = Q1 / torch.sum(Q1, dim=1, keepdim=True)

        Q2 = 1.0 / (1.0 + dist2 / v)
        Q2 = Q2.pow((v + 1.0) / 2.0)
        Q2 = Q2 / torch.sum(Q2, dim=1, keepdim=True)

        P = (Q**2) / torch.sum(Q, dim=1, keepdim=True)
        P = P / torch.sum(P, dim=1, keepdim=True)

        # Loss is Kl-divergence between P and Q
        kl_div_loss = torch.sum(P * torch.log(P / ((Q + Q1 +Q2) / 3)))
        # get the mse between the data and the embedding 1 
        mse_loss = torch.mean((x_emb1 - x)**2)

        entropy_loss = entropy_loss + entropy_loss1 + entropy_loss2
        link_loss = link_loss + link_loss1 + link_loss2

        return x_emb1 , S , kl_div_loss , x_cluster , adj , x_emb2 , x_cluster1 , adj1 , x_cluster2 , adj2 , Q , Q1 , Q2 , link_loss , entropy_loss , mse_loss

        

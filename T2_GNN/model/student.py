import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv

import os

# Setting environment variable to debug CUDA errors
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class Student(nn.Module):
    def __init__(self, nbr_nodes: int, in_channels: int, out_channels: int,
                 hid_channels: int, feat_hidd: int, str_hidd: int, dropout: float,
                 device: torch.device, teta: float = 0.5, tau: float = 0.5) -> None:
        """
        Initialize the Student model.
        """
        super(Student, self).__init__()

        # Initialize various parameters and layers of the Student model
        self.nbr_nodes = nbr_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.device = device
        self.tau = tau
        self.hid_channels = hid_channels
        self.feat_hidd = feat_hidd
        self.str_hidd = str_hidd
        self.raw_teta = nn.Parameter(torch.tensor(teta, dtype=torch.float32).to(self.device))
        
        self.gcn1 = DenseSAGEConv(in_channels, hid_channels)
        self.gcn2 = DenseSAGEConv(hid_channels, hid_channels)
        self.gcn3 = DenseSAGEConv(hid_channels, out_channels)
        self.link_predictor = nn.Linear(out_channels, 1, bias=True)

        self.feat_student = nn.Linear(self.feat_hidd, self.hid_channels, bias=True)
        self.str_student = nn.Linear(self.str_hidd, self.hid_channels, bias=True)
        self.to(self.device)

    def forward(self, X: torch.Tensor, Adj: torch.Tensor):
        """
        Forward pass for the Student model.
        """
        # Handle missing values in X
        imp = torch.zeros((X.shape[0], self.in_channels)).to(self.device)
        X = torch.where(torch.isnan(X), imp, X).to(self.device)

        # Compute intermediate node embeddings and store them
        middle_representation = []
        h1 = self.gcn1(X, Adj)
        middle_representation.append(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h1 = F.leaky_relu(h1)
        
        h2 = self.gcn2(h1, Adj)
        middle_representation.append(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        h2 = F.leaky_relu(h2)
        
        h3 = self.gcn3(h2, Adj)
        middle_representation.append(h3)

        # Link prediction
        link_prediction = self.link_predictor(h3)

        # Compute teta value
        teta = torch.sigmoid(self.raw_teta)

        return h3, middle_representation, link_prediction, teta

    def sim(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute the similarity between two sets of embeddings.
        """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute the semi-supervised loss.
        """
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        bet_sim = f(self.sim(z1, z2))

        return -torch.log(
            bet_sim.diag() / (refl_sim.sum(1) + bet_sim.sum(1) - refl_sim.diag())
        )
    def loss(self , z_student : torch.tensor , z_struct : torch.tensor , z_feat : torch.tensor , mean:bool = True):
        Emb_student_1 = z_student[0].squeeze(0)
        Emb_student_2 = z_student[1].squeeze(0)
        Emb_student_3 = z_student[2].squeeze(0)

        Emb_struct_1 = z_struct[0].squeeze(0)
        Emb_struct_2 = z_struct[1].squeeze(0)
        Emb_struct_3 = z_struct[2].squeeze(0)

        Emb_feat_1 =  z_feat[0].squeeze(0)
        Emb_feat_2 =  z_feat[1].squeeze(0)
        Emb_feat_3 =  z_feat[2].squeeze(0)


        feat_stu_1 = self.semi_loss(Emb_student_1 , Emb_feat_1)
        feat_stu_2 = self.semi_loss(Emb_student_2 , Emb_feat_2)
        feat_stu_3 = self.semi_loss(Emb_student_3 , Emb_feat_3)

        struct_stu_1 = self.semi_loss(Emb_student_1 , Emb_struct_1)
        struct_stu_2 = self.semi_loss(Emb_student_2 , Emb_struct_2)
        struct_stu_3 = self.semi_loss(Emb_student_3 , Emb_struct_3)

        del Emb_student_1 , Emb_student_2 , Emb_student_3 , Emb_struct_1 , Emb_struct_2 , Emb_struct_3 , Emb_feat_1 , Emb_feat_2 , Emb_feat_3

        feat_struct_1 = feat_stu_1.mean() if mean else feat_stu_1.sum()
        feat_struct_2 = feat_stu_2.mean() if mean else feat_stu_2.sum()
        feat_struct_3 = feat_stu_3.mean() if mean else feat_stu_3.sum()

        struct_struct_1 = struct_stu_1.mean() if mean else struct_stu_1.sum()
        struct_struct_2 = struct_stu_2.mean() if mean else struct_stu_2.sum()
        struct_struct_3 = struct_stu_3.mean() if mean else struct_stu_3.sum()

        loss_feat = feat_struct_1 + feat_struct_2 + feat_struct_3
        loss_struct = struct_struct_1 + struct_struct_2 + struct_struct_3

        del feat_stu_1 , feat_stu_2 , feat_stu_3 , struct_stu_1 , struct_stu_2 , struct_stu_3

        return loss_feat , loss_struct

        

        


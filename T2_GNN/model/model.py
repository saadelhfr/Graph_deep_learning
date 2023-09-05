import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv , DenseGINConv

class BaseModule(nn.Module):
    def __init__(self, device):
        super(BaseModule, self).__init__()
        self.device = device
        self.to(device)
    
    def weights_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight.data, mode='fan_in', nonlinearity='leaky_relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias.data)

class TeacherEdge(BaseModule):
    def __init__(self, config, device):
        super(TeacherEdge, self).__init__(device)
        
        # Parameters from config
        self.nbr_nodes = config["nbr_nodes"]
        self.in_channels = config["in_channels_edge"]
        self.hid_channels = config["hid_channels_edge"]
        self.out_channels = config["out_channels_edge"]
        self.dropout = config["dropout_edge"]

        # Define layers
        self.gcn1 = DenseSAGEConv(self.in_channels, self.hid_channels)
        self.gcn2 = DenseSAGEConv(self.hid_channels, self.hid_channels)
        self.gcn3 = DenseSAGEConv(self.hid_channels, self.out_channels)
        self.linear = nn.Linear(self.nbr_nodes, self.in_channels, bias=True)

    def forward(self, Adj, pe_feat, X):
        x = self.linear(pe_feat)
        mask = torch.isnan(X)
        X[mask] = x[mask]

        return self._forward_gcn(X, Adj)

    def _forward_gcn(self, x, Adj):
        middle_representation = []
        for gcn in [self.gcn1, self.gcn2, self.gcn3]:
            x = gcn(x, Adj)
            middle_representation.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.leaky_relu(x)
        
        return x, middle_representation

class TeacherFeatures(BaseModule):
    def __init__(self, config, device):
        super(TeacherFeatures, self).__init__(device)
        
        # Parameters from config
        self.nbr_nodes = config["nbr_nodes"]
        self.in_channels = config["in_channels_feat"]
        self.hid_channels = config["hid_channels_feat"]
        self.out_channels = config["out_channels_feat"]
        self.dropout = config["dropout_feat"]

        # Feature initialization
        self.imp_features = nn.Parameter(torch.empty(size=(self.nbr_nodes, self.in_channels)))
        nn.init.xavier_uniform_(self.imp_features.data, gain=1.414)

        # Layers
        self.linear1 = nn.Linear(self.in_channels, self.hid_channels)
        self.linear2 = nn.Linear(self.hid_channels, self.hid_channels)
        self.linear3 = nn.Linear(self.hid_channels, self.out_channels)
        
        # Initialize weights
        self.weights_init()

    def forward(self, x, pe_feat):
        idx = pe_feat._indices()[1]
        imp_feat_reduced = self.imp_features[idx]
        nan_mask = torch.isnan(x)
        x[nan_mask] = imp_feat_reduced[nan_mask]
        x.to(self.device)
        
        return self._forward_linear(x)

    def _forward_linear(self, x):
        middle_representation = []
        for linear in [self.linear1, self.linear2, self.linear3]:
            x = linear(x)
            middle_representation.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.leaky_relu(x)
        
        return x, middle_representation


class Student(nn.Module):
    def __init__(self, nbr_nodes, in_channels, out_channels, dropout, device, 
                 hid_channels_list=[64, 64, 64], tau=0.5, teta_init=0.5):
        """
        Initialize the Student class.

        Args:
        - nbr_nodes (int): Number of nodes.
        - in_channels (int): Input channels.
        - out_channels (int): Output channels.
        - dropout (float): Dropout rate.
        - device (torch.device): Device.
        - hid_channels_list (list): List containing hidden dimensions for GCN layers.
        - tau (float): Tau value for semi_loss.
        - teta_init (float): Initial value for teta.
        """
        super(Student, self).__init__()
        
        self.dropout = dropout
        self.device = device
        self.tau = tau
        self.raw_teta = nn.Parameter(torch.tensor(teta_init, dtype=torch.float32).to(device))

        # Assuming len(hid_channels_list) gives us the number of GCN layers required
        self.gcn_layers = nn.ModuleList()
        input_dim = in_channels
        for hid_dim in hid_channels_list:
            self.gcn_layers.append(DenseSAGEConv(input_dim, hid_dim))
            input_dim = hid_dim

        self.link_predictor = nn.Linear(input_dim, 1, bias=True)
        self.to(device)

    def forward(self, X, Adj):
        X = torch.where(torch.isnan(X), torch.zeros_like(X).to(self.device), X).to(self.device)

        middle_representations = []
        for layer in self.gcn_layers:
            X = layer(X, Adj)
            middle_representations.append(X)
            X = F.dropout(X, p=self.dropout, training=self.training)
            X = F.leaky_relu(X)

        link_prediction = self.link_predictor(X)
        teta = torch.sigmoid(self.raw_teta)
        
        return X, middle_representations, link_prediction, teta

    @staticmethod
    def sim(z1, z2):
        """Compute similarity between two tensors."""
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1, z2):
        """Compute semi loss."""
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        bet_sim = f(self.sim(z1, z2))
        return -torch.log(bet_sim.diag() / (refl_sim.sum(1) + bet_sim.sum(1) - refl_sim.diag()))

    def loss(self, z_student, z_struct, z_feat, mean=True):
        losses_feat, losses_struct = [], []

        for z_s, z_s_t, z_f_t in zip(z_student, z_struct, z_feat):
            # If dimensions of student and teacher representations differ, adapt the student's dimensions.
            if z_s.shape[-1] != z_f_t.shape[-1]:
                transform_layer = nn.Linear(z_s.shape[-1], z_f_t.shape[-1]).to(self.device)
                z_s = transform_layer(z_s)
                
            loss_feat = self.semi_loss(z_s.squeeze(0), z_f_t.squeeze(0))

            # If dimensions of student and structure representations differ, adapt the student's dimensions.
            if z_s.shape[-1] != z_s_t.shape[-1]:
                transform_layer = nn.Linear(z_s.shape[-1], z_s_t.shape[-1]).to(self.device)
                z_s = transform_layer(z_s)

            loss_struct = self.semi_loss(z_s.squeeze(0), z_s_t.squeeze(0))

            if mean:
                loss_feat = loss_feat.mean()
                loss_struct = loss_struct.mean()

            losses_feat.append(loss_feat)
            losses_struct.append(loss_struct)

        total_loss_feat = sum(losses_feat)
        total_loss_struct = sum(losses_struct)

        return total_loss_feat, total_loss_struct


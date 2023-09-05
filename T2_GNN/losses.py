import torch 
import torch.nn.functional as F 
from torch.nn import KLDivLoss
from torch_geometric.utils import negative_sampling




def kl_divergence(P , Q , mean : bool = True):
    """
    Computes the KL divergence between P and Q
    """
    if mean :
        return torch.mean(torch.sum(P * torch.log(P/Q) , dim=1)) 
    else :    
        return torch.sum(torch.sum(P* torch.log(P/Q) , dim=1)) 
    
def js_divergence(P , Q , mean : bool = True):
    M = 0.5 * (P + Q)
    if mean :
        return 0.5 * kl_divergence(P , M , mean) + 0.5 * kl_divergence
    else :
        return 0.5 * kl_divergence(P , M , mean) + 0.5 * kl_divergence
    
def link_prediction(X1 , X2 ):
    X = X1 * X2
    X = torch.sigmoid(X)
    return X
    
def link_prediction_objectif(X , edge_index):
    """
    X of shaoe (nbr_nodes , dimension)
    edge_index of shape (2 , nbr_edges)
    """
    nbr_nodes = X.shape[0]

    nbr_edges = edge_index.shape[1]
    pos_edge_index = edge_index
    negativ_edge_index = negative_sampling(edge_index , nbr_nodes , nbr_edges , method="sparse")
    pos_pred = link_prediction(X[pos_edge_index[0]] , X[pos_edge_index[1]])
    neg_pred = link_prediction(X[negativ_edge_index[0]] , X[negativ_edge_index[1]])

    pos_weight = 3
    neg_weight = 7

    pos_loss = -torch.log(pos_pred + 1e-15)
    neg_loss = -torch.log(1 - neg_pred + 1e-15)
    loss = pos_weight*torch.mean(pos_loss) + neg_weight*torch.mean(neg_loss)


    del pos_loss , neg_loss , pos_pred , neg_pred
    return loss







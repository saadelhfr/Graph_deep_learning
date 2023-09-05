import torch 
import torch.nn.functional as F
import networkx as nx
import numpy as np 
from scipy.sparse import csr_matrix
from .PPR_Matrix.ppr import topk_ppr_matrix


def students_t_kernel_euclidean(z1 : torch.tensor , z2 : torch.tensor , v=1.0):
    """
    Computes the students t kernel between z1 and z2
    we consider that the two tensors are of shape 
    (N , n_features)
    where N can be different for z1 and z2

    """
    z1 = z1.unsqueeze(1)
    z2 = z2.unsqueeze(0)
    
    Q = 1.0/ (1.0 + torch.sum(torch.pow(z1-z2 , 2) , dim=2)/v)
    Q = Q.pow((v+1.0)/2.0)
    Q = Q / torch.sum(Q, dim=1, keepdim=True)  
    return Q


def student_t_kernel_cosine(z1 : torch.tensor , z2 : torch.tensor , v=1.0):
    """
    Computes the students t kernel between z1 and z2
    we consider that the two tensors are of shape 
    (N , n_features)
    where N can be different for z1 and z2

    """
    z1 = F.normalize(z1 , p=2 , dim=1)
    z2 = F.normalize(z2 , p=2 , dim=1)

    z1 = z1.unsqueeze(1)
    z2 = z2.unsqueeze(0)

    cosine_sim = 1 - torch.sum(z1*z2 , dim=-1)

    Q = 1.0/ (1.0 + torch.sum((1-cosine_sim) , dim=-1)/v )

    Q = Q.pow((v+1.0)/2.0)
    Q = Q/torch.sum(Q , dim=1 , keepdim=True)


    return Q
def generate_targer_distribution(Q):
    """
    Generates the target distribution for the student t kernel  

    """

    P = (Q**2) / torch.sum(Q, dim=1, keepdim=True)
    P = P / torch.sum(P , dim=1 , keepdim=True)
    return P

"""
relabel the nodes of the graph into indexes from 0 to N-1
"""

def relabel_nodes(graph):
    set_of_nodes = set(graph.nodes())
    for edge in graph.edges():
        set_of_nodes.update(edge)
    new_labels = [ i for i in range(len(set_of_nodes))]
    map = {old: new for old, new in zip(set_of_nodes, new_labels)}
    return nx.relabel_nodes(graph, map , copy=True)

"""
get adjacency matrix from a graph
"""

def get_adjacency_matrix(graph):
    return nx.adjacency_matrix(graph).todense()

def get_adjacency_matrix_torch(graph):
    return torch.tensor(get_adjacency_matrix(graph)).float()

"""
get augmented adjacency matrix from a graph using PPR_Matrix module 
"""

def get_augmented_adjacency_matrix(graph , alpha  = 0.15, epsilon = 0.1 , topk = 10):
    sparse_adjacency_matrix = get_adjacency_matrix(graph)
    idx = np.arange(sparse_adjacency_matrix.shape[0])
    A_sparse = csr_matrix(sparse_adjacency_matrix)

    P = topk_ppr_matrix(A_sparse, alpha, epsilon, idx, topk)

    P = P.toarray()
    P= torch.tensor(P)
    return P

    

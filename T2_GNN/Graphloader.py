import torch 
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data ,  DataLoader  , Dataset
import networkx as nx
import random 
import os 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from .utils import get_adjacency_matrix_torch , get_augmented_adjacency_matrix
from tqdm import tqdm
import pickle 

random.seed(42)


"""
Takes as input a networkx Graoh and returns a list of subgraphs based on a sampling strategy 
the goal is to be able to calculate the adjancency matrix and the PPR matrix without problems with the Ram memory 



"""


class GraphLoader(Dataset):
    def __init__(self , load_path , sampling_strategy, device , nbr_subgraphs_merged = None ,  save_path :str ="T2_GNN/Subgraphs/"  ,directed :bool = False) :
        super().__init__()
        self.load_path = load_path
        self.save_path = save_path
        self.sampling_strategy = sampling_strategy
        self.directed = directed
        self.device = device
        self.list_subgraphs_paths :list[str] = []
        self.nbr_subgraphs_merged = nbr_subgraphs_merged


    def len(self):
        return len(self.list_subgraphs_paths)

    def get(self , idx: int):
        data =  torch.load(self.list_subgraphs_paths[idx])
        return data


    def load_nx_graph(self): 
        with open(self.load_path, "rb") as f:
            graph = picke.load(f)
        graph = graph.to_undirected()
        self.graph = graph

        self.nbr_nodes = self.graph.number_of_nodes()

    def save_torch_geo_subgraph(self , subgraph , path : str) :
        torch.save(subgraph , path)

    def torch_graph_from_set(self , set , nodes_before  ): 
        subgraph = self.graph.subgraph(set)
        adj = get_adjacency_matrix_torch(subgraph)
        adj_aug = get_augmented_adjacency_matrix(subgraph)
        I = torch.FloatTensor(torch.eye(subgraph.number_of_nodes()))
        nbr_zerosleft = nodes_before
        nbr_zerosright = self.nbr_nodes - subgraph.number_of_nodes() - nbr_zerosleft
        I = torch.cat((torch.zeros(( subgraph.number_of_nodes() ,nbr_zerosleft)) , I , torch.zeros((  subgraph.number_of_nodes() , nbr_zerosright))) , dim=1)
        data = from_networkx(subgraph)
        data.adj = adj
        data.pe_feat = I.to_sparse()
        data.adj_aug = adj_aug
        data.x=data.attr_dict
        nodes_before = nodes_before + subgraph.number_of_nodes()
        return data , nodes_before


    def sample_subgraphs(self):

        if self.sampling_strategy == "connected_components" : 
            assert self.directed == False , "Connected components sampling strategy is only available for undirected graphs"
            self.subgraphs = list(nx.connected_components(self.graph))
            self.nbr_nodes_subgraphs = [len(subgraph) for subgraph in self.subgraphs]
            self.nbr_subgraphs = len(self.subgraphs)

        else : 
            raise NotImplementedError("Sampling strategy not implemented")
    
    def get_subgraphs_set(self , nbr_subgraphs_merged : int =300 ,  save : bool = True) : 
        """
    
        """
        if self.nbr_subgraphs_merged is not None :
            nbr_subgraphs_merged = self.nbr_subgraphs_merged
        
        list_subgraphs  = self.subgraphs.copy()
        random.shuffle(list_subgraphs)
        connected_components_sublists = [list_subgraphs[i:i + nbr_subgraphs_merged] for i in range(0, len(list_subgraphs), nbr_subgraphs_merged)]
        connected_components_sublists_merged = [set().union(*connected_components_sublist) for connected_components_sublist in connected_components_sublists]
        return connected_components_sublists_merged


    def prepare(self):
        print("Loading graph ....")
        self.load_nx_graph()
        print("finished loading graph")
        print("Sampling subgraphs ....")
        self.sample_subgraphs()
        print("finished sampling subgraphs")
        print("Merging subgraphs ....")
        self.subgraph_sets = self.get_subgraphs_set()  # Save sets of merged subgraphs
        print("finished merging subgraphs")

    def save_all(self):
        nodes_before = 0
        for i, subgraph_set in tqdm(enumerate(self.subgraph_sets) , total=len(self.subgraph_sets)):
            subgraph , nodes_before = self.torch_graph_from_set(subgraph_set , nodes_before)
            path = os.path.join(self.save_path, f"subgraph_{i}.pth")
            self.list_subgraphs_paths.append(path)
            self.save_torch_geo_subgraph(subgraph, path)
            del subgraph

    def load_all(self):
        self.subgraphs = [torch.load(path) for path in self.list_subgraphs_paths]










        



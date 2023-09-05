An implementation of multiple methods for Graph deep learning on trees. The methods are implemented in PyTorch and are based on the following papers:

- [HGNN] : https://arxiv.org/abs/2012.09600 this is just multy layers of dfcn where each time we reduce the number of predicted clusters (HGNN is hsort for hierarchical graph neural network) 
- [T2-GNN] : https://arxiv.org/abs/2212.12738 with modification for clustering and edge prediction (T2-GNN is because we use two teachers one for the structure and one for the node attributes and a student model that is trained with knowledge distillation)
- [Tree_transformers] : https://arxiv.org/abs/2010.11929 based on the vision transfrmer architecture with bert ecoder as the transformer for the biderectional encoding of the sequences


## Installation 

run the following command to install the required packages

```pip install -r requirements.txt```

## Usage

The project is set up as a library so you can use jupyter notebook for training and testing the models and build on your own implementations or models 


## Data 

The data used and benchmarks established are proprietary information and sadly cannot be shared.

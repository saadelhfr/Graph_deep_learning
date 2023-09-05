import torch 
import math
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from tqdm import tqdm
import numpy as np






def train(model , device , nbr_epochs ,data  ): 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in tqdm(range(nbr_epochs)):
        optimizer.zero_grad()
        x = data.x.to(device)
        adj = data.adj.to(device)
        x_emb1 , S , kl_div_loss , x_cluster , adj , x_emb2 , x_cluster1 , adj1 , x_cluster2 , adj2 , Q , Q1 , Q2 , link_loss , entropy_loss , mse_loss = model(data)
        

        loss = kl_div_loss + link_loss + entropy_loss #+ mse_loss
        print('{} loss: {}'.format(epoch, loss))
        loss.backward()

        optimizer.step()


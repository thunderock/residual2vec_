# -*- coding: utf-8 -*-
# @Author: Ashutosh Tiwari
# @Email: checkashu@gmail.com
# @Date:   2023-02-14 19:46:23
# @Filepath: dataset/copenhagen_data.py

from os.path import join as j
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import download_url, extract_zip



class CopenhagenData(object):
    def __init__(self, root='/tmp/') -> None:
        edges = 'https://figshare.com/ndownloader/files/13389320'
        nodes = 'https://figshare.com/ndownloader/files/13389440'
        edges = download_url(edges, root)
        nodes = download_url(nodes, root)
        edges = pd.read_csv(edges)
        nodes = pd.read_csv(nodes)
        max_node = edges.max().max()
        
        # remove nodes not in edges
        edge_index = torch.tensor(edges.values.T, dtype=torch.long)
        # remove self loop
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
        node_exists_mask = torch.zeros((max_node + 1,), dtype=torch.bool)
        node_exists_mask[edge_index.flatten().long()] = True
        old_idx_to_new_idx = torch.zeros((max_node + 1,), dtype=torch.long)
        usable_nodes = node_exists_mask.nonzero().flatten()
        old_idx_to_new_idx[usable_nodes] = torch.arange(usable_nodes.size(0))
        edge_index = old_idx_to_new_idx[edge_index]
        
        y = np.full((max_node + 1,), 2, dtype=np.int32)
        y[nodes["# user"]] = nodes.female
        y = y[usable_nodes]
        self.group_col = torch.from_numpy(y)
        self.edge_index = torch.unique(torch.cat([edge_index, edge_index.flip(0)], dim=1), dim=1)
    
    def get_grouped_col(self):
        return self.group_col
        
        
        
        
        
        
            
    
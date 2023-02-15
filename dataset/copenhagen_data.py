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
        
        y = np.full((max_node + 1,), 2, dtype=np.int32)
        y[nodes["# user"]] = nodes.female
        self.group_col = torch.from_numpy(y)
        edge_index = torch.tensor(edges.values.T, dtype=torch.long)
        self.edge_index = torch.unique(torch.cat([edge_index, edge_index.flip(0)], dim=1), dim=1)
    
    def get_grouped_col(self):
        return self.group_col
        
        
        
        
        
        
            
    
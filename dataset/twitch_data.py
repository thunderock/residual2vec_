# -*- coding: utf-8 -*-
# @Author: Ashutosh Tiwari
# @Email: checkashu@gmail.com
# @Date:   2023-02-16 15:56:43
# @Filepath: dataset/twich_data.py


from os.path import join as j
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import download_url, extract_zip
from sklearn.preprocessing import LabelEncoder

class TwitchData(object):
    
    def __init__(self, root='/tmp/', group_col='mature') -> None:
        edges = 'files/large_twitch_edges.csv'
        nodes = 'files/large_twitch_features.csv'
        features = pd.read_csv(nodes)
        features['language'] = LabelEncoder().fit_transform(features['language'])
        self.X = features[['mature', 'language']]
        
        edges = pd.read_csv(edges)
        edge_index = torch.tensor(edges.values.T, dtype=torch.long)
        # there are no self loops and no reverse edges
        self.edge_index = torch.unique(torch.cat([edge_index, edge_index.flip(0)], dim=1), dim=1)
        self.group_col = group_col
        del edges, edge_index, features
        
    def get_grouped_col(self):
        return torch.from_numpy(self.X[self.group_col].values)
        
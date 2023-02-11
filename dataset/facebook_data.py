# -*- coding: utf-8 -*-
# @Author: Ashutosh Tiwari
# @Email: checkashu@gmail.com
# @Date:   2023-02-10 19:47:52
# @Filepath: dataset/facebook_data.py

import torch
from torch_geometric.datasets import LINKXDataset
import pandas as pd

class FacebookData(object):
    def __init__(self, group_col:str = "gender", root:str = '/tmp/') -> None:
        dataset = LINKXDataset(root=root, name="Penn94")[0]
        self.group_col = dataset.y
        df = pd.DataFrame({"gender": dataset.y.numpy()})
        edge_index = dataset.edge_index
        # repeat the edges to make it undirected
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        self.edge_index = edge_index
        self.X = dataset.x
    
    def get_grouped_col(self):
        return self.group_col
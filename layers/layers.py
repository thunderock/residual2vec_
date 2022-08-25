# @Filename:    layers.py.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        8/19/22 2:01 AM

import torch
from torch_geometric.nn import GCNConv, GATConv

class GCNConvMean(GCNConv):
    def forward(self, x, edge_index, **kwargs):

        X = super().forward(x, edge_index, **kwargs)
        return torch.mean(X, dim=0)

class GATConvMean(GATConv):
    def forward(self, x, edge_index, **kwargs):

        X = super().forward(x, edge_index, **kwargs)
        return torch.mean(X, dim=0)
# -*- coding: utf-8 -*-
# @Author: Ashutosh Tiwari
# @Email: checkashu@gmail.com
# @Date:   2023-04-12 16:59:33
# @Filepath: dataset/generic_data.py
import numpy as np
import torch
class GenericData(object):
    def __init__(self, edge_index, group_membership):
        assert edge_index.shape[0] == 2, "edge_index should be of shape (2, num_edges)"
        if type(edge_index) == np.ndarray:
            edge_index = torch.from_numpy(edge_index)
        if type(group_membership) == np.ndarray:
            group_membership = torch.from_numpy(group_membership)
        self.edge_index = edge_index
        self.group_col = group_membership
    
    def get_grouped_col(self):
        return self.group_col
# -*- coding: utf-8 -*-
# @Author: Ashutosh Tiwari
# @Email: checkashu@gmail.com
# @Date:   2023-02-19 17:05:02
# @Filepath: utils/network_splitter.py
import numpy as np
import torch
from scipy import sparse
from scipy.sparse import csgraph, coo_matrix
from tqdm import tqdm
from torch_sparse import SparseTensor
from torch_sparse.tensor import from_scipy


class NetworkTrainTestSplitter(object):
    def __init__(self, num_nodes: int, edge_list: torch.Tensor, fraction=0.5):
        """
        :param G: Networkx graph object. Origin Graph
        :param fraction: Fraction of edges that will be removed (test_edge).
        """
        # # dont need graph
        # self.G = G
        # # always undirected
        # self.directed = directed
        self.num_nodes = num_nodes
        self.original_edge_set = edge_list
        self.total_number_of_edges = len(self.original_edge_set[0])
        self.number_of_test_edges = int(self.total_number_of_edges * fraction)

        self.test_edges = None
        self.train_edges = None

    def save_splitted_result(self, path_train, path_test):
        torch.save(self.train_edges, path_train)
        torch.save(self.test_edges, path_test)


class NetworkTrainTestSplitterWithMST(NetworkTrainTestSplitter):
    def __init__(self, num_nodes: int, edge_list: torch.Tensor, fraction=0.5):
        """Only support undirected Network.
        :param G: Networkx graph object. Origin Graph
        :param fraction: Fraction of edges that will be removed (test_edge).
        """
        # remove duplicate edges from undirected graph
        edge_list = torch.unique(torch.cat([edge_list, edge_list.flip(0)], dim=1), dim=1)
        edge_list = edge_list[:, edge_list[0] < edge_list[1]]
        
        super(NetworkTrainTestSplitterWithMST, self).__init__(
            num_nodes=num_nodes, edge_list=edge_list, fraction=fraction
        )

    def _get_edge_list(self, adj: SparseTensor) -> torch.Tensor:
        row, col, _ = adj.coo()
        return torch.stack([row, col], dim=0)

    def find_mst(self, edge_list):
        row, col = edge_list
        adj = SparseTensor(row=row, col=col, sparse_sizes=(self.num_nodes, self.num_nodes)).to("cpu")
        # adj = adj.to_symmetric()
        MST = csgraph.minimum_spanning_tree(adj.to_scipy())
        # why isn't mst symmetric
        return from_scipy(MST).to("cpu").to_symmetric(), adj

    def train_test_split(self):
        mst_adj, adj = self.find_mst(self.original_edge_set)
        # very slow need to do this in torch
        not_mst_adj = adj.to_scipy() - mst_adj.to_scipy()
        not_mst_edge_index = self._get_edge_list(from_scipy(not_mst_adj))
        mst_edge_list = self._get_edge_list(mst_adj)
        print("MST edge list shape: ", mst_edge_list.shape)
        print("Not MST edge list shape: ", not_mst_edge_index.shape)
        if len(not_mst_edge_index[0]) < self.number_of_test_edges:
            raise Exception(
                "Cannot remove edges by keeping the connectedness. Decrease the `fraction` parameter"
            )

        edge_ids = torch.from_numpy(np.random.choice(
            len(not_mst_edge_index[0]), self.number_of_test_edges, replace=False
        ))
        self.test_edges = not_mst_edge_index[:, edge_ids]

        self.train_edges = torch.cat((not_mst_edge_index[:, ~edge_ids], mst_edge_list), dim=1)
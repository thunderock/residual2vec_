# @Filename:    weighted_node2vec.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        8/31/22 10:29 AM
from utils.config import *
import torch
from torch_geometric.nn.models import Node2Vec
from torch_sparse import SparseTensor
from residual2vec import random_walk_sampler
random_walk = torch.ops.torch_cluster.random_walk


class WeightedNode2Vec(Node2Vec):
    def __init__(self, edge_weights, num_nodes, group_membership, edge_index, **params):
        Node2Vec.__init__(self, edge_index=edge_index, **params)
        self.weights = edge_weights
        # self.adj = SparseTensor.from_edge_index(edge_index=edge_index, edge_attr=edge_weights, sparse_sizes=(num_nodes, num_nodes))
        # self.adj = self.adj.to('cpu')
        adj = self.adj.to_symmetric()
        row, col, _ = adj.coo()
        from scipy import sparse
        ones = np.ones(row.shape[0], dtype=np.int32)
        A = sparse.csr_matrix((ones, (row.numpy(), col.numpy())), shape=(num_nodes, num_nodes))
        from utils import graph
        G = graph.from_numpy(A, undirected=True)
        G.attr = group_membership
        graph.set_weights(G, exp_=2., p_bndry=.7, l=2)
        self.A = graph.edge_weights_to_sparse(G, A)
        self.sampler = random_walk_sampler.RandomWalkSampler(self.A)

    def pos_sample(self, batch):
        batch = batch.repeat(self.walks_per_node)
        rowptr, col, _ = self.adj.csr()
        rw = random_walk(rowptr, col, batch, self.walk_length, self.p, self.q)
        # print(rw[0].shape, rw[1].shape)
        if not isinstance(rw, torch.Tensor):
            rw = rw[0]

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)
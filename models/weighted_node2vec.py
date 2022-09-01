# @Filename:    weighted_node2vec.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        8/31/22 10:29 AM
from utils.config import *
from utils import graph
import torch
from torch_geometric.nn.models import Node2Vec
from scipy import sparse
from residual2vec import random_walk_sampler
random_walk = torch.ops.torch_cluster.random_walk


class WeightedNode2Vec(Node2Vec):
    def __init__(self, num_nodes, group_membership, edge_index, weighted_adj=None, **params):
        """
        :param weighted_adj: can be a weighted sparse matrix or its path
        """
        Node2Vec.__init__(self, num_nodes=num_nodes, edge_index=edge_index, **params)
        if isinstance(group_membership, torch.Tensor):
            self.group_membership = group_membership.numpy()

        if weighted_adj:
            if isinstance(weighted_adj, sparse.csr_matrix):
                self.weighted_adj = weighted_adj
            else:
                self.weighted_adj = sparse.load_npz(weighted_adj)

        else:
            adj = self.adj.to_symmetric()
            row, col, _ = adj.coo()
            ones = np.ones(row.shape[0], dtype=np.int32)
            A = sparse.csr_matrix((ones, (row.numpy(), col.numpy())), shape=(num_nodes, num_nodes))
            G = graph.from_numpy(A, undirected=True)
            G.attr = group_membership
            graph.set_weights(G, exp_=2., p_bndry=.7, l=2)
            # TODO (ashutiwa): cross walk doesn't produce symmetric matrix for undirected graph
            self.weighted_adj = graph.edge_weights_to_sparse(G, A)
        self.sampler = random_walk_sampler.RandomWalkSampler(adjmat=self.weighted_adj, walk_length=self.walk_length + 1, q=self.q, p=self.p)
        self.edge_index = edge_index

    def pos_sample(self, batch):
        batch = batch.repeat(self.walks_per_node)
        rowptr, col, _ = self.adj.csr()
        rw = self.sampler.sampling(start=batch.numpy())
        # print(rw[0].shape, rw[1].shape, rowptr.shape, col.shape, batch.shape)
        # print(rw.shape)
        rw = torch.from_numpy(rw)
        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)
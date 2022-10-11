# @Filename:    weighted_node2vec.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        8/31/22 10:29 AM
import numpy as np

from utils.config import *
from utils import graph
import torch
from scipy import sparse
from node2vec.node2vecs import GensimNode2Vec
from torch_sparse import SparseTensor
from torch_sparse.tensor import from_scipy

class Node2Vec(object):
    def __init__(self, embedding_dim):
        self.node2vec = GensimNode2Vec(vector_size=embedding_dim)

    def train_and_get_embs(self, save=None):
        self.node2vec.fit(self.adj)
        embs = self.node2vec.transform()
        if save is not None:
            np.save(save, embs)
        return embs


class WeightedNode2Vec(Node2Vec):
    def __init__(self, num_nodes, group_membership, embedding_dim, edge_index=None, weighted_adj=None):
        """
        :param weighted_adj: can be a weighted sparse matrix or its path
        : Only call this when you have a weighted_adj, i.e. crosswalk
        """

        Node2Vec.__init__(self, embedding_dim=embedding_dim)
        if isinstance(group_membership, torch.Tensor):
            self.group_membership = group_membership.numpy()

        if not weighted_adj:
            assert edge_index is not None
            row, col = edge_index
            adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes)).to_symmetric()
            row, col, _ = adj.coo()
            ones = np.ones(row.shape[0], dtype=np.int32)
            A = sparse.csr_matrix((ones, (row.numpy(), col.numpy())), shape=(num_nodes, num_nodes))
            G = graph.from_numpy(A, undirected=True)
            G.attr = group_membership
            graph.set_weights(G, exp_=2., p_bndry=.7, l=2)
            # TODO (ashutiwa): cross walk doesn't produce symmetric matrix for undirected graph
            adj = graph.edge_weights_to_sparse(G, A)
            adj = from_scipy(adj)
        else:
            adj = from_scipy(sparse.load_npz(weighted_adj))
        self.adj = adj.to_scipy()


class UnWeightedNode2Vec(Node2Vec):
    def __init__(self, num_nodes, embedding_dim, weighted_adj, edge_index):
        if not weighted_adj:
            assert edge_index is not None
            row, col = edge_index
            adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
        else:
            adj = from_scipy(sparse.load_npz(weighted_adj))
        Node2Vec.__init__(self, embedding_dim=embedding_dim)
        self.adj = adj.to_symmetric().to_scipy()




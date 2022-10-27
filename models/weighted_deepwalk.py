# @Filename:    weighted_deepwalk.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        8/31/22 10:29 AM
import numpy as np

from utils.config import *
from utils import graph
import torch
from scipy import sparse
from torch_sparse import SparseTensor
from torch_sparse.tensor import from_scipy
from graph_embeddings import DeepWalk as dw, Fairwalk as fw

class DeepWalk(object):
    def __init__(self, embedding_dim):

        self.deepwalk = dw()
        self.embedding_dim = embedding_dim

    def train_and_get_embs(self, save=None):
        assert sparse.issparse(self.adj)
        deepwalk = self.deepwalk.fit(self.adj)
        embs = deepwalk.transform(self.embedding_dim)
        if save is not None:
            np.save(save, embs)
        self.deepwalk = deepwalk
        return embs


class CrossWalkDeepWalk(DeepWalk):
    def __init__(self, num_nodes, group_membership, embedding_dim, edge_index=None, weighted_adj=None):
        """
        :param weighted_adj: can be a weighted sparse matrix or its path
        : Only call this when you have a weighted_adj, i.e. crosswalk
        """

        DeepWalk.__init__(self, embedding_dim=embedding_dim)
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
        elif sparse.issparse(weighted_adj):
            adj = from_scipy(weighted_adj)
        else:
            adj = from_scipy(sparse.load_npz(weighted_adj))
        self.adj = adj.to_scipy()


class UnWeightedDeepWalk(DeepWalk):
    def __init__(self, num_nodes, embedding_dim, weighted_adj, edge_index):
        if not weighted_adj:
            assert edge_index is not None
            row, col = edge_index
            adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
        elif sparse.issparse(weighted_adj):
            adj = from_scipy(weighted_adj)
        else:
            adj = from_scipy(sparse.load_npz(weighted_adj))
        DeepWalk.__init__(self, embedding_dim=embedding_dim)
        self.adj = adj.to_symmetric().to_scipy()


class FairWalkDeepWalk(DeepWalk):
    def __init__(self, num_nodes, group_membership, embedding_dim, edge_index=None, weighted_adj=None):
        """
        :param weighted_adj: can be a weighted sparse matrix or its path
        : Only call this when you have a weighted_adj, i.e. crosswalk
        """

        DeepWalk.__init__(self, embedding_dim=embedding_dim)
        if isinstance(group_membership, torch.Tensor):
            self.group_membership = group_membership.numpy()

        if not weighted_adj:
            assert edge_index is not None
            row, col = edge_index
            adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes)).to_symmetric()
            row, col, _ = adj.coo()
            ones = np.ones(row.shape[0], dtype=np.int32)
            A = sparse.csr_matrix((ones, (row.numpy(), col.numpy())), shape=(num_nodes, num_nodes))
            fairwalk = fw(group_membership=group_membership)
            adj = fairwalk._get_adj_matrix(A)
        elif sparse.issparse(weighted_adj):
            adj = weighted_adj
        else:
            adj = sparse.load_npz(weighted_adj)
        self.adj = adj

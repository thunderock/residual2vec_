# @Filename:    weighted_node2vec.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        8/31/22 10:29 AM
import numpy as np

from utils import snakemake_utils
import torch
from scipy import sparse
from node2vec.node2vecs import GensimNode2Vec
from torch_sparse import SparseTensor
from torch_sparse.tensor import from_scipy

class Node2Vec(object):
    def __init__(self, embedding_dim):
        self.node2vec = GensimNode2Vec(vector_size=embedding_dim)

    def train_and_get_embs(self, save=None):
        assert sparse.issparse(self.adj)
        self.node2vec.fit(self.adj)
        embs = self.node2vec.transform()
        if save is not None:
            np.save(save, embs)
        return embs


class WeightedNode2Vec(Node2Vec):
    def __init__(self, num_nodes,group_membership, embedding_dim, weighted_adj):
        """
        :param weighted_adj: can be a weighted sparse matrix or its path
        : Only call this when you have a weighted_adj, i.e. crosswalk
        """

        Node2Vec.__init__(self, embedding_dim=embedding_dim)
        if isinstance(group_membership, torch.Tensor):
            self.group_membership = group_membership.numpy()


        if sparse.issparse(weighted_adj):
            adj = from_scipy(weighted_adj)
        else:
            adj = from_scipy(sparse.load_npz(weighted_adj))
        self.adj = adj.to_scipy()


class UnWeightedNode2Vec(Node2Vec):
    def __init__(self, num_nodes, embedding_dim, weighted_adj):

        if sparse.issparse(weighted_adj):
            adj = from_scipy(weighted_adj)
        else:
            adj = from_scipy(sparse.load_npz(weighted_adj))
        Node2Vec.__init__(self, embedding_dim=embedding_dim)
        self.adj = adj.to_scipy()


class FairWalkNode2Vec(Node2Vec):
    def __init__(self, num_nodes, group_membership, embedding_dim,weighted_adj):
        """
        :param weighted_adj: can be a weighted sparse matrix or its path
        : Only call this when you have a weighted_adj, i.e. crosswalk
        """

        Node2Vec.__init__(self, embedding_dim=embedding_dim)
        if isinstance(group_membership, torch.Tensor):
            self.group_membership = group_membership.numpy()


        if sparse.issparse(weighted_adj):
            adj = weighted_adj
        else:
            adj = sparse.load_npz(weighted_adj)
        self.adj = adj


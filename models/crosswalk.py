# @Filename:    crosswalk.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        8/6/22 8:47 PM
import numpy as np

from graph_embeddings import Node2Vec
from graph_embeddings.utils import to_adjacency_matrix
from residual2vec import random_walk_sampler
from utils import graph

class Crosswalk(Node2Vec):
    def __init__(self, group_membership, window_length, num_walks, alpha=.7, exp=2., **params):
        Node2Vec.__init__(self, num_walks=num_walks, window_length=window_length, **params)
        self.group_membership = group_membership
        self.w2vparams = {
            "sg": 1,
            "hs": 1,
            "min_count": 0,
            "workers": 4,
        }
        self.alpha = alpha
        self.window_length = window_length
        self.num_walks = num_walks
        self.exp = exp

    def fit(self, net):
        A = to_adjacency_matrix(net)
        # get new A here using the crosswalk paper and then use sampler to get walks

        G = graph.from_numpy(A, undirected=True)
        G.attr = self.group_membership
        # better way to count number of classes
        cnt_classes = np.unique(self.group_membership, ).shape[0]
        graph.set_weights(G, exp_=self.exp, p_bndry=self.alpha, l=cnt_classes)
        A = graph.edge_weights_to_sparse(G, A)
        # choose start point randomly here

        nodes = np.arange(A.shape[0])
        # shuffles in place
        np.random.shuffle(nodes)
        nodes = nodes[:self.num_walks]

        self.sampler = random_walk_sampler.RandomWalkSampler(A)
        self.sampler.walks = self.sampler.sampling(start=nodes)
        self.sampler.window_length = self.window_length
        return self

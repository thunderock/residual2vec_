# @Filename:    crosswalk.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        8/6/22 8:47 PM
import numpy as np

from graph_embeddings import Fairwalk, Node2Vec
from graph_embeddings.utils import to_adjacency_matrix
from residual2vec import random_walk_sampler

class Crosswalk(Node2Vec):
    def __init__(self, group_membership, alpha=.7, exp=2., **params):
        Node2Vec.__init__(self, **params)
        self.group_membership = group_membership
        self.w2vparams = {
            "sg": 1,
            "hs": 1,
            "min_count": 0,
            "workers": 4,
        }
        self.sampler = random_walk_sampler.RandomWalkSampler
        self.alpha = alpha
        self.exp = exp
        self.num_classes = np.unique()

    def fit(self, net):
        A = to_adjacency_matrix(net)
        # get new A here using the crosswalk paper and then use sampler to get walks


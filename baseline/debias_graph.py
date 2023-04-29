# -*- coding: utf-8 -*-
# @Author: Ashutosh Tiwari
# @Email: checkashu@gmail.com
# @Date:   2023-01-30 12:07:40
# @Filepath: graph_debiaswe/debias_graph.py

# assumptions for converting to graph
# gender specific words: node ids of group specific words, probably the one closest to centroids
# defitional pairs: pairs of centroids of groups
# equalize words: these should be equidistant from the centroids of the groups, so these are the most distant pairs

import numpy as np
from baseline.we_utils import get_direction, EMB_UTILS
from baseline import we
from sklearn.decomposition import PCA

def debias_wrapper(emb, equalize, direction):
    embs = emb.copy()
    nodes, dim = embs.shape
    # K = np.unique(y).shape[0]
    # direction = get_direction(embs, y, direction_method)
    # direction = we.doPCA(definitional, embs, num_components=1).components_[0]
    # gender specific words are the node ids, lets have a vector of size 1x nodes
    # where i == true denotes that it is gender specific
    # definitional are not the node ids, but the centroids of the groups
    # equalize are pairs of node ids
#     assert definitional.shape == (K, K - 1)
    # for i in range(nodes):
    #     group = y[i]
    #     if drop_gender_specific_words:
    #         embs[i] = we.drop(embs[i], direction)

    #     else:
    #         if i not in gender_specific_words[group]:
    #             embs[i] = we.drop(embs[i], direction)

    embs = EMB_UTILS.normalize(embs)

    # need to change this to take just one node and apply the equation from paper
    
    for a in equalize:
        
        w_b = direction * np.dot(embs[a], direction)
        embs[a] = embs[a] - w_b
    embs = EMB_UTILS.normalize(embs)
    return embs

    

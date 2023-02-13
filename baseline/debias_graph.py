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

def debias_wrapper(emb, gender_specific_words, definitional, 
        equalize, y, direction, drop_gender_specific_words=False):
    embs = emb.copy()
    nodes, dim = embs.shape
    K = np.unique(y).shape[0]
    # direction = get_direction(embs, y, direction_method)
    # direction = we.doPCA(definitional, embs, num_components=1).components_[0]
    # gender specific words are the node ids, lets have a vector of size 1x nodes
    # where i == true denotes that it is gender specific
    # definitional are not the node ids, but the centroids of the groups
    # equalize are pairs of node ids
#     assert definitional.shape == (K, K - 1)
    assert direction.shape == (dim, )
    assert equalize.shape[1:] == (2, )
    assert gender_specific_words.shape[0] == K


    for i in range(nodes):
        group = y[i]
        if drop_gender_specific_words:
            embs[i] = we.drop(embs[i], direction)

        else:
            if i not in gender_specific_words[group]:
                embs[i] = we.drop(embs[i], direction)

    embs = EMB_UTILS.normalize(embs)

    for (a,b) in equalize:
        a, b = int(a), int(b)
        y = we.drop((embs[a] + embs[b]) / 2, direction)
        import warnings
        # print y value in case of warning
        z = np.sqrt(1 - np.linalg.norm(y)**2)

        if (embs[a] - embs[b]).dot(direction) < 0:
            z = -z
        embs[a] = y + z * direction
        embs[b] = y - z * direction

    embs = EMB_UTILS.normalize(embs)

    return embs

    

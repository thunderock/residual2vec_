# -*- coding: utf-8 -*-
# @Author: Ashutosh Tiwari
# @Email: checkashu@gmail.com
# @Date:   2023-01-30 16:12:09
# @Filepath: graph_debiaswe/utils.py


import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def get_direction(embs, y=None, method='PCA'):
    if method == 'PCA':
        pca = PCA(n_components=1)
        pca.fit(embs)
        direction = pca.components_[0]
    elif method == 'LDA':
        assert y is not None, "y should be provided for LDA"
        direction = LinearDiscriminantAnalysis(n_components=1).fit(embs, y).coef_[0]
    else:
        assert False, "method should be either PCA or LDA"
    return direction


def doPCA(pairs, embs, num_components=1):
    matrix = []
    # these are column vectors for each class
    tuple_size, size = pairs.shape
    # print("size: ", size, "tuple_size: ", tuple_size)
    for i in range(size):
        center = np.mean(embs[pairs[:, i]], axis=0)
        # print(center.shape, pairs[:, i].shape)
        for member in range(tuple_size):
            matrix.append(embs[pairs[member, i]] - center)
        # matrix.append(embs[pairs[:, i]] - center)
    
    matrix = np.vstack(matrix)
    pca = PCA(n_components = num_components)
    pca.fit(matrix)
    return pca


class EMB_UTILS(object):

    @staticmethod
    def normalize(embs):
        norm = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / norm

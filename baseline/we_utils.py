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



class EMB_UTILS(object):

    @staticmethod
    def normalize(embs):
        norm = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / norm

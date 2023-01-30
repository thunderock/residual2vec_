# -*- coding: utf-8 -*-
# @Author: Ashutosh Tiwari
# @Email: checkashu@gmail.com
# @Date:   2023-01-29 18:38:17
# @Filepath: bolukbasi/wrapper.py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def get_gender_direction(embs, method, labels=None):
    if method == 'pca':
        pca = PCA(n_components=1)
        pca.fit(embs)
        return pca.components_[0]
    if method == 'lda':
        assert labels is not None, "Labels are required for LDA."
        lda = LDA(n_components=1)
        lda.fit(embs, labels)
        return lda.coef_[0]
    else:
        assert False, "Unknown method."

def get_equalize_embs(embs, n):
    # figure out the most distant pairs, use cosine similarity
    return embs[:n]

def debias_embeddings(embs: np.array, gender_direction: str, equalize: str, y: np.array=None):
    gender_direction = get_gender_direction(embs=embs, method=gender_direction, labels=y)
    

    
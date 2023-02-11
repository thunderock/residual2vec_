# -*- coding: utf-8 -*-
# @Author: Ashutosh Tiwari
# @Email: checkashu@gmail.com
# @Date:   2023-02-01 13:36:12
# @Filepath: models/fast_knn_cpu.py
import numpy as np
from scipy import sparse
import faiss


class FastKnnCpu(object):
    
    def __init__(self, k=1, metric="cosine", exact=None, nprobe=50, min_cluster_size=10000) -> None:
        
        # exact can be True or False. If None, then it is automatically set to True if n_samples < 1000
        self.k = k
        self.metric = metric
        self.exact = exact
        self.nprobe = nprobe
        self.min_cluster_size = min_cluster_size
        assert self.metric in ["cosine"], "Only cosine distance is supported, because farthest is also supported"
    
    def fit(self, X):
        n_samples, n_features = X.shape[0], X.shape[1]
        X = X.astype("float32")
        if self.exact is None:
            if n_samples < 1000:
                self.exact = True
            else:
                self.exact = False
        index = faiss.IndexFlatIP(n_features)
        if not self.exact:
            nlist = np.maximum(int(n_samples / self.min_cluster_size), 2)
            faiss_metric = faiss.METRIC_INNER_PRODUCT
            index = faiss.IndexIVFFlat(index, n_features, int(nlist), faiss_metric)
        if not index.is_trained:
            Xtrain = X[
                np.random.choice(
                    X.shape[0],
                    np.minimum(X.shape[0], self.min_cluster_size * 5),
                    replace=False,
                ),
                :,
            ].copy(order="C")
            index.train(Xtrain)
        index.add(X)
        index.nprobe = self.nprobe
        self.index = index
        return self
    
    def predict(self, X, farthest=False, return_distance=False):
        X = X.astype("float32")
        if farthest:
            X = -X
        D, I = self.index.search(X, k=self.k)
        if return_distance:
            return I, D
        return I
    


# @Filename:    fast_knn.py.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        9/7/22 10:43 AM

"""k-nearest neighbor predictor"""
import numpy as np
from scipy import sparse
import faiss


class FastkNN:
    def __init__(self, k=5, metric="euclidean", exact=True, gpu_id=None):
        self.k = k
        self.metric = metric
        self.gpu_id = gpu_id
        self.exact = exact

    # fit knn model
    def fit(self, X):
        # make knn graph
        X = self._homogenize(X)
        self.n_indexed_samples = X.shape[0]
        self._make_faiss_index(X)
        return self

    def predict(self, X):

        X = self._homogenize(X)

        return self._make_knn_graph(X, self.k, exclude_selfloop=False)

    def _make_faiss_index(self, X):
        n_samples, n_features = X.shape[0], X.shape[1]
        X = X.astype("float32")
        if n_samples < 1000:
            self.exact = True

        index = (
            faiss.IndexFlatL2(n_features)
            if self.metric == "euclidean"
            else faiss.IndexFlatIP(n_features)
        )

        if not self.exact:
            # code_size = 32
            train_sample_num = np.minimum(100000, X.shape[0])
            nlist = int(np.ceil(np.sqrt(train_sample_num)))
            faiss_metric = (
                faiss.METRIC_L2
                if self.metric == "euclidean"
                else faiss.METRIC_INNER_PRODUCT
            )

            index = faiss.IndexIVFFlat(index, n_features, nlist, faiss_metric)

        if self.gpu_id is not None:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, self.gpu_id, index)

        if not index.is_trained:
            Xtrain = X[
                np.random.choice(X.shape[0], train_sample_num, replace=False), :
            ].copy(order="C")
            index.train(Xtrain)

        index.add(X)
        self.index = index

    def _make_knn_graph(self, X, k, exclude_selfloop=True, weighted=False):
        _, indices = self.index.search(X.astype("float32"), k)
        return indices

    def _homogenize(self, X, Y=None):
        if self.metric == "cosine":
            X = np.einsum("ij,i->ij", X, 1 / np.maximum(np.linalg.norm(X, axis=1), 1e-32))
        X = X.astype("float32")

        if X.flags["C_CONTIGUOUS"]:
            X = X.copy(order="C")

        if Y is not None:
            if sparse.issparse(Y):
                if not sparse.isspmatrix_csr(Y):
                    Y = sparse.csr_matrix(Y)
            elif isinstance(Y, np.ndarray):
                Y = sparse.csr_matrix(Y)
            else:
                raise ValueError("Y must be a scipy sparse matrix or a numpy array")
            Y.data[Y.data != 1] = 1
            return X, Y
        else:
            return X
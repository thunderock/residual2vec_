# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-01-17 04:25:23
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-01-19 02:49:31
# %%
import faiss
import numpy as np
import pandas as pd
from scipy import sparse
import sys
from scipy.stats import entropy
from tqdm import tqdm

if "snakemake" in sys.modules:
    emb_file = snakemake.params["emb_file"]
    node_table_file = snakemake.input["node_table_file"]
    output_file = snakemake.output["output_file"]
else:
    emb_file = "../data/derived/preprocessed/airport/airport_one/airport_gat_None_node2vec_r2v_embs.npy"
    #emb_file = "../data/derived/preprocessed/airport/airport_one/airport_gat_None_deepwalk_deepwalk_embs.npy"
    node_table_file = "../data/derived/preprocessed/airport/node_table.csv"

# ========================
# Load
# ========================
emb = np.load(emb_file)
node_table = pd.read_csv(node_table_file)

# ========================
# Calculating the fairness
# ========================

def make_faiss_index(
    X, metric, gpu_id=None, exact=False, nprobe=50, min_cluster_size=10000
):
    """Create an index for the provided data
    :param X: data to index
    :type X: numpy.ndarray
    :raises NotImplementedError: if the metric is not implemented
    :param metric: metric to calculate the similarity. euclidean or cosine.
    :type mertic: string
    :param gpu_id: ID of the gpu, defaults to None (cpu).
    :type gpu_id: string or None
    :param exact: exact = True to find the true nearest neighbors. exact = False to find the almost nearest neighbors.
    :type exact: boolean
    :param nprobe: The number of cells for which search is performed. Relevant only when exact = False. Default to 10.
    :type nprobe: int
    :param min_cluster_size: Minimum cluster size. Only relevant when exact = False.
    :type min_cluster_size: int
    :return: faiss index
    :rtype: faiss.Index
    """
    n_samples, n_features = X.shape[0], X.shape[1]
    X = X.astype("float32")
    if n_samples < 1000:
        exact = True

    index = (
        faiss.IndexFlatL2(n_features)
        if metric == "euclidean"
        else faiss.IndexFlatIP(n_features)
    )

    if not exact:
        nlist = np.maximum(int(n_samples / min_cluster_size), 2)
        faiss_metric = (
            faiss.METRIC_L2 if metric == "euclidean" else faiss.METRIC_INNER_PRODUCT
        )
        index = faiss.IndexIVFFlat(index, n_features, int(nlist), faiss_metric)

    if gpu_id != "cpu":
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, gpu_id, index)

    if not index.is_trained:
        Xtrain = X[
            np.random.choice(
                X.shape[0],
                np.minimum(X.shape[0], min_cluster_size * 5),
                replace=False,
            ),
            :,
        ].copy(order="C")
        index.train(Xtrain)
    index.add(X)
    index.nprobe = nprobe
    return index

# Get the group membership
group_cols = np.unique(node_table["group_id"].values, return_inverse=True)[1]

# Number of groups
K = int(np.max(group_cols) + 1)

# Number of nodes
N = emb.shape[0]

# Normalize to calculate the cosine distance
emb = np.einsum("ij,i->ij", emb, 1 / np.maximum(1e-12,np.linalg.norm(emb, axis=1)))

# If the number of nodes is too large, we sample some nodes and calculate the fairness for that nodes
n_max_nodes = 10000
if emb.shape[0] > n_max_nodes:
    sampled_nodes = np.random.choice(N, n_max_nodes, replace=False)
    Nk = n_max_nodes
else:
    sampled_nodes = np.arange(N)
    Nk = N

# Number of neighbors
klist = [5, 10, 50, 100, 500, 1000]

# Find kmax-nearest neighbors.
kmax = int(np.max([k for k in klist if (k +1) < N]))
index = make_faiss_index(emb, gpu_id = "cpu", metric = "cosine")
Dist, Nids = index.search(emb[sampled_nodes, :].astype("float32"), k = kmax + 1)
Nids = Nids[:, 1:]

results = []
for k in tqdm(klist):
    if (k+1)>N:
        break
    nids = Nids[:, :k]
    r, c = np.outer(np.arange(nids.shape[0]), np.ones(nids.shape[1])).reshape(-1), group_cols[nids].reshape(-1)
    z = r + 1j * c
    z, freq = np.unique(z, return_counts=True)
    r, c = np.real(z).astype(int), np.imag(z).astype(int)
    N = emb.shape[0]

    D = sparse.csr_matrix((freq, (r, c)), shape=(Nk, K)).toarray()
    denom = np.array(np.sum(D, axis =1)).reshape(-1)
    D = np.einsum("ij,i->ij", D, 1/denom)

    Dref = np.bincount(group_cols)
    Dref= Dref/np.sum(Dref)
    Dref = np.outer(np.ones(Nk), Dref)
    ent = np.mean(np.array(entropy(D, qk = Dref, axis = 1)).reshape(-1))
    results.append({"k":k, "relativeEntropy":ent})
pd.DataFrame(results).to_csv(output_file, index=False)
# %%

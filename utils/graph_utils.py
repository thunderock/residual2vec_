# @Filename:    graph_utils.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        8/8/22 1:21 AM
import numpy as np
import pandas as pd
from scipy import sparse
from utils.config import DEVICE, CUDA, GPU_ID
import networkx as nx
from tqdm import tqdm, trange
import faiss

def get_edge_df(G: nx.Graph):
    """
    Get the edge list from the graph
    :param G: graph
    :return: edge list
    """
    num_edges = G.number_of_edges()
    source = np.zeros(num_edges, dtype=np.int32)
    target = np.zeros(num_edges, dtype=np.int32)
    for i, (u, v) in enumerate(tqdm(G.edges())):
        source[i] = u
        target[i] = v
    return pd.DataFrame({"source": source, "target": target})


def reconstruct_graph(emb, n, m):
    """
    emb: embedding matrix n * d
    n: number of nodes
    m: number of top edges to pick
    """
    # choose top m edges to reconstruct the graph
    S = emb @ emb.T
    S = np.triu(S, k=1)
    r, c, v = sparse.find(S)
    idx = np.argsort(-v)[:m]
    r, c, v = r[idx], c[idx], v[idx]
    B = sparse.csr_matrix((v, (r, c)), shape=(n, n))
    B = B + B.T
    B.data = B.data * 0 + 1
    return get_edge_df(nx.from_scipy_sparse_matrix(B + B.T))


def get_edges_faiss(emb, k=10, batch_size=2000, weight=False, metric=faiss.METRIC_INNER_PRODUCT, exact=False):
    """
    emb: embedding matrix n * d
    k: number of top edges to pick for every node
    batch_size: number of nodes to process at a time
    weight: if True, return the weight of the edges
    """
    assert not (weight and exact), "Not implemented yet"
    assert CUDA, "this implementation is only for CUDA"
    n_nodes, embedding_size = emb.shape

    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatIP(embedding_size)

    train_sample_size = np.minimum(1000000, n_nodes)
    nlist = int(np.ceil(np.sqrt(train_sample_size)))

    index = faiss.IndexIVFFlat(index, embedding_size, nlist, metric)
    index = faiss.index_cpu_to_gpu(res, GPU_ID, index)
    index.add(emb.copy())
    targets = np.array([index.search(i, k=k) for i in tqdm(np.array_split(emb, n_nodes // batch_size))])
    # if weight:
    #     # dont use this right now, not normalized, only returns distance for now
    #     weights = np.concatenate([i[0].flatten() for i in targets])
    # else:
    #     weights = np.ones(n_nodes * k)
    targets = np.concatenate([i[1].flatten() for i in targets])
    return pd.DataFrame({
        "source": np.repeat(np.arange(n_nodes), k),
        "target": targets,
        # "weight": weights
    })
# @Filename:    graph_utils.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        8/8/22 1:21 AM
import numpy as np
import pandas as pd
from scipy import sparse
from utils.config import GPU_ID, DISABLE_TQDM
import networkx as nx
from tqdm import tqdm


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


def get_edges_fastknn_faiss(emb, k=10, batch_size=2000):
    """
    emb: embedding matrix n * d
    k: number of top edges to pick for every node
    batch_size: number of nodes to process at a time
    weight: if True, return the weight of the edges
    """
    from models.fast_knn import FastkNN
    assert GPU_ID >= 0, "only implemented for GPU for time being"
    n_nodes, embedding_size = emb.shape
    knn = FastkNN(k=k, metric='cosine', exact=False, gpu_id=GPU_ID).fit(emb)

    targets = np.concatenate([knn.predict(i).flatten() for i in tqdm(np.array_split(emb, n_nodes // batch_size), disable=DISABLE_TQDM)])
    return pd.DataFrame({
        "source": np.repeat(np.arange(n_nodes), k),
        "target": targets,
    })
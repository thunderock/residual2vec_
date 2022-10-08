# @Filename:    graph_utils.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        8/8/22 1:21 AM
import numpy as np
import pandas as pd
from scipy import sparse
from utils.snakemake_utils import get_torch_sparse_from_edge_index
from utils.config import GPU_ID, DISABLE_TQDM
import networkx as nx
from tqdm import tqdm
import torch
from tqdm import trange

def _negative_sampling_sparse(edge_index, n_nodes, n_neg_samples=None, iter_limit=1000, return_pos_samples=False):
    # removing duplicated edges (because of symmetry
    edge_index = torch.unique(torch.sort(edge_index, dim=0).values, dim=1)
    n_neg_samples = edge_index.size(1) if n_neg_samples is None else n_neg_samples
    adj = get_torch_sparse_from_edge_index(edge_index, n_nodes).to_scipy(layout='csr')
    neg_adj = get_torch_sparse_from_edge_index(torch.empty((2, 0), dtype=torch.long), n_nodes).to_scipy(layout='csr')
    nodes = torch.concat((edge_index[0], edge_index[1])).numpy()
    iterations, sampled = 0, 0
    while sampled < n_neg_samples and iterations < iter_limit:
        remaining = n_neg_samples - sampled
        # choosing start nodes with replacement
        start_nodes = nodes[np.random.randint(0, n_nodes, (remaining,))]
        # choosing end nodes with replacement
        end_nodes = nodes[np.random.randint(0, n_nodes, (remaining,))]
        # removing self loops and start less than end
        mask = (start_nodes != end_nodes) & (start_nodes < end_nodes)
        start_nodes = start_nodes[mask]
        end_nodes = end_nodes[mask]

        # removing edges that are already present in the graph
        mask = np.array((adj[start_nodes, end_nodes] == 0).flat)
        start_nodes = start_nodes[mask]
        end_nodes = end_nodes[mask]

        # removing edges in negative_edges
        mask = np.array((neg_adj[start_nodes, end_nodes] == 0).flat)
        start_nodes = start_nodes[mask]
        end_nodes = end_nodes[mask]

        # adding these to negative edges
        neg_adj[start_nodes, end_nodes] = 1
        # should be symmetric
        neg_adj[end_nodes, start_nodes] = 1
        sampled = neg_adj.nnz // 2
        iterations += 1

    neg_edge_index = torch.from_numpy(np.vstack(neg_adj.nonzero())).long()
    # this will contain duplicates because of symmetry
    neg_edge_index = torch.unique(torch.sort(neg_edge_index, dim=0).values, dim=1)
    if return_pos_samples:
        return neg_edge_index, edge_index[:, :neg_edge_index.size(1)]
    return neg_edge_index

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


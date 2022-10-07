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

def _pair(s, t):
    k = s + t
    return (k * (k + 1) * .5 + torch.minimum(s, t)).long()

def _depair(k):
    w = torch.floor((torch.sqrt(8 * k + 1) - 1) / 2)
    t = (w ** 2 + w) / 2
    y = k - t
    x = w - y
    return torch.vstack((x.long(), y.long()))


def negative_sampling(edge_index, n_nodes, n_neg_samples=None, iter_limit=1000, return_pos_samples=False, use_adj=False):
    """
    This edge index leads to a symmetric matrix for sure
    """
    # comment this before merging in master
    # assert get_torch_sparse_from_edge_index(edge_index, n_nodes).is_symmetric()
    if use_adj:
        return _negative_sampling(edge_index, n_nodes, n_neg_samples, iter_limit, return_pos_samples)
    neg_edges = torch.tensor([], dtype=torch.long)
    # not flattening the edge_index because source and targets are already same
    anonymized_edges = torch.unique(_pair(edge_index[0], edge_index[1]))
    nodes = edge_index[0]
    iterations = 0
    n_neg_samples = anonymized_edges.size(0) if n_neg_samples is None else n_neg_samples

    while neg_edges.size(0) < n_neg_samples and iterations < iter_limit:
        remaining = n_neg_samples - neg_edges.size(0)
        # choosing start nodes with replacement
        start_nodes = nodes[torch.randint(0, n_nodes, (remaining,))]
        # choosing end nodes with replacement
        end_nodes = nodes[torch.randint(0, n_nodes, (remaining,))]
        # removing self loops
        mask = start_nodes != end_nodes
        start_nodes = start_nodes[mask]
        end_nodes = end_nodes[mask]
        # concatenating start and end nodes
        paired = _pair(start_nodes, end_nodes)
        # removing edges that are already present in the graph
        paired = paired[~torch.isin(paired, anonymized_edges)]

        # removing edges in negative_edges
        paired = paired[~torch.isin(paired, neg_edges)]

        # adding these to negative edges
        neg_edges = torch.cat((neg_edges, paired))
        iterations += 1

    if return_pos_samples:
        return _depair(neg_edges), _depair(anonymized_edges[:neg_edges.size(0)])
    return _depair(neg_edges)

def _negative_sampling(edge_index, n_nodes, n_neg_samples=None, iter_limit=1000, return_pos_samples=False):
    # removing duplicated edges (because of symmetry
    edge_index = torch.unique(torch.sort(edge_index, dim=0).values, dim=1)
    adj = get_torch_sparse_from_edge_index(edge_index, n_nodes).to_dense().bool()
    neg_edges = torch.zeros_like(adj)
    nodes = torch.concat((edge_index[0], edge_index[1]))
    iterations = 0
    n_neg_samples = edge_index.size(1) if n_neg_samples is None else n_neg_samples
    sampled = 0
    while sampled < n_neg_samples and iterations < iter_limit:
        # choosing start nodes with replacement
        start_nodes = nodes[torch.randint(0, n_nodes, (n_neg_samples,))]
        # choosing end nodes with replacement
        end_nodes = nodes[torch.randint(0, n_nodes, (n_neg_samples,))]
        # removing self loops and start less than end
        mask = (start_nodes != end_nodes) & (start_nodes < end_nodes)
        start_nodes = start_nodes[mask]
        end_nodes = end_nodes[mask]

        # removing edges that are already present in the graph
        mask = adj[start_nodes, end_nodes] == 0
        start_nodes = start_nodes[mask]
        end_nodes = end_nodes[mask]

        # removing edges in negative_edges
        mask = neg_edges[start_nodes, end_nodes] == 0
        start_nodes = start_nodes[mask]
        end_nodes = end_nodes[mask]

        # adding these to negative edges
        neg_edges[start_nodes, end_nodes] = 1
        sampled += start_nodes.size(0)
        iterations += 1

    neg_edges = neg_edges.nonzero(as_tuple=True)
    if return_pos_samples:
        return torch.stack(neg_edges), edge_index[:, :neg_edges[0].size(0)]
    return torch.stack(neg_edges)


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

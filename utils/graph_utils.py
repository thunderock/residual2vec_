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
from tqdm import tqdm, trange
import torch
from heapq import heappush, heappop, heapify

def get_farthest_pairs(embs, y, metric="cosine", same_class=True, per_class_count=1):
    """returns the farthest pairs of nodes in the graph
    Returns:
        np.array: farthest pairs of nodes
    """
    assert metric in ["cosine"], "only cosine distance is supported"
    n = len(y)
    uy, y = np.unique(y, return_inverse=True)
    K = len(uy)
    ret = np.zeros((K * per_class_count, 2), dtype=np.int64)

    from models.fast_knn_cpu import FastKnnCpu

    for k in range(K):
        reverse_mapping = np.zeros(n, dtype=np.int64)
        idx = np.where(y == k)[0]
        train_idx = idx if same_class else np.where(y != k)[0]
        reverse_mapping[np.arange(len(train_idx))] = train_idx
        knn = FastKnnCpu(k=per_class_count,).fit(embs[train_idx])
        # select k farthest points
        X = knn.predict(X=embs[idx], farthest=True, return_distance=True)
        ids, dist = X[0].flatten(), X[1].flatten()
        ids = reverse_mapping[ids]
        pairs = np.stack([np.repeat(idx, per_class_count), ids], axis=1)
        # select the farthest distance
        most_distant = np.argsort(dist)[:per_class_count]
        ret[k * per_class_count: (k + 1) * per_class_count] = pairs[most_distant]

    return ret
            

    

def get_n_nearest_neighbors_for_nodes(nodes, embs, k=1, metric="cosine"):
    assert nodes.shape[1] == embs.shape[1]
    n_nodes, emb_dim = embs.shape
    from scipy.spatial.distance import cdist
    x = cdist(nodes, embs, metric=metric)
    targets = np.zeros((len(nodes), k), dtype=np.int64)
    for i in range(len(nodes)):
        targets[i, :] = np.argsort(x[i, :])[:k]
    return targets

def get_centroid_per_group(emb: np.array, y: np.array):
    """returns K * D matrix where K is the number of classes and D is the dimension of the embedding

    Args:
        emb (np.array): embedding matrix (N * D)
        y (np.array): labels
    """
    n = len(y)
    uy, y = np.unique(y, return_inverse=True)
    K = len(uy)
    centroids = np.zeros((K, emb.shape[1]))
    for k in range(K):
        centroids[k] = emb[y == k].mean(axis=0)
    return centroids



def _negative_sampling_sparse(edge_index, n_nodes, n_neg_samples=None, iter_limit=1000, return_pos_samples=False):
    # removing duplicated edges (because of symmetry
    edge_index = torch.unique(torch.sort(edge_index, dim=0).values, dim=1)
    n_neg_samples = edge_index.size(1) if n_neg_samples is None else n_neg_samples
    adj = get_torch_sparse_from_edge_index(edge_index, n_nodes).to_scipy(layout='csr')
    # we only need the upper triangular part of the adjacency matrix
    neg_adj = sparse.triu(get_torch_sparse_from_edge_index(torch.empty((2, 0), dtype=torch.long), n_nodes).to_scipy(layout='csr', dtype=torch.bool), k=1, format='lil')

    nodes = torch.concat((edge_index[0], edge_index[1])).numpy()
    iterations, sampled = 0, 0
    while sampled < n_neg_samples and iterations < iter_limit:
        remaining = n_neg_samples - sampled
        # choosing start nodes with replacement
        end_nodes = nodes[np.random.randint(0, n_nodes, (remaining,))]
        # choosing end nodes with replacement
        
        start_nodes = np.random.choice(n_nodes, (remaining,))

        # removing self loops and start less than end
        # this doesn't change the fact that sampling is proportional to degree because all these are bidirectional edges
        end_nodes, start_nodes = np.maximum(start_nodes, end_nodes), np.minimum(start_nodes, end_nodes)
        mask = start_nodes != end_nodes
        start_nodes = start_nodes[mask]
        end_nodes = end_nodes[mask]

        # pick unique pairs here
        mask = np.unique(np.stack((start_nodes, end_nodes)), axis=1, return_index=True)[1]
        start_nodes = start_nodes[mask]
        end_nodes = end_nodes[mask]

        # optional randomize start and end nodes across axis=1
        # removing edges that are already present in the graph
        mask = adj[start_nodes, end_nodes] == 0
        if mask.size:
            mask = np.array(mask.flat)
            start_nodes = start_nodes[mask]
            end_nodes = end_nodes[mask]

        # removing edges in negative_edges, taking != 0 because it is faster than == 0
        mask = neg_adj[start_nodes, end_nodes] != 0
        if mask.size:
            # convert matrix to numpy array
            mask = np.squeeze(np.asarray(mask.todense()))
            start_nodes = start_nodes[~mask]
            end_nodes = end_nodes[~mask]
        # adding these to negative edges
        neg_adj[start_nodes, end_nodes] = 1
        # should be symmetric, dont need this because start nodes are always smaller than end nodes
        # neg_adj[end_nodes, start_nodes] = 1
        # only fill upper triangular part, therefore total number of edges is number of ones
        sampled = neg_adj.nnz
        iterations += 1
    # return neg_adj
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


def generate_embedding_with_word2vec(A, dim, noise_sampler, device, outfile=None, learning_rate=1e-3, batch_size=256,):
    from node2vec import node2vecs
    n_nodes = A.shape[0]
    model = node2vecs.Word2Vec(
        vocab_size=n_nodes + 1,
        embedding_size=dim,
        padding_idx=n_nodes,
        learn_outvec=False, # Set True to learn out vector. Otherwise, the out-vector is copied from in-vector.
    )
    model = model.to(device)
    noise_sampler.fit(A) # Train the noise sampler
    
    # We sample triplet of nodes containing one negative sample per positive sample.
    loss_func = node2vecs.Node2VecTripletLoss(n_neg=1)
    dataset = node2vecs.TripletDataset(
        adjmat=A,
        window_length=1,
        noise_sampler=noise_sampler,
        padding_id=n_nodes,
        buffer_size=256,
        context_window_type="double",
        epochs=1,
        negative=1,
        p = 1,
        q = 1, 
        num_walks = 20,
        walk_length = 80,
    )
    # Generate embedding 
    node2vecs.train(
        model=model,
        dataset=dataset,
        loss_func=loss_func,
        batch_size=batch_size,
        device=device,
        learning_rate=learning_rate,
        num_workers=20,
        outputfile=outfile
    )
    model.eval()
    
    # Retrieve the embedding vector. We use the in-vector. 
    return model.ivectors.weight.data.cpu().numpy()[:n_nodes, :]



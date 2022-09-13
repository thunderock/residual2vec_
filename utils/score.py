# @Filename:    score.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        7/29/22 9:46 PM
import numpy as np
import networkx as nx
import pandas as pd
from scipy import sparse
from tqdm import trange

def accuracy(y_pred, y_true):
    return y_pred.max(1)[1].eq(y_true).double().mean().numpy()

def equal_opportunity_sparse(sp_mt, y):
    pass


def _get_preds_for_parity_score(edges, class_label, n_nodes, y):
    y_pred = pd.Series(np.zeros(n_nodes), dtype=np.int32)
    for priv_node in (y == class_label).nonzero()[0]:
        y_pred[priv_node] = 1
        # print(edges[edges.source == priv_node].target.values)
        y_pred[edges[edges.source == priv_node].target.values] = 1
    return y_pred

def _get_preds_for_opportunity_score(edges, class_label, n_nodes, y):
    y_pred = pd.Series(np.zeros(n_nodes), dtype=np.int32)
    for priv_node in (y == class_label).nonzero()[0]:
        y_pred[priv_node] = 1
        # print(edges[edges.source == priv_node].target.values)
        y_pred[edges[edges.source == priv_node].target.values] = 1
    return y_pred

def opportunity_difference(edges, y):
    """
    edges: edge df with source and target column
    y: original labels
    """
    assert isinstance(y, np.ndarray) and isinstance(edges, pd.DataFrame)
    classes, counts = np.unique(y, return_counts=True)
    # for each class figure out all the neighbors
    n_nodes = y.shape[0]
    # check if these are labels
    assert np.alltrue(y >= 0) and np.alltrue(y < counts.shape[0]) and y.dtype in [np.int64, np.int32]
    n_classes = len(classes)
    scores = np.empty(n_classes, dtype=np.float32)
    for i in range(n_classes):
        y_class = np.zeros(n_nodes, dtype=np.int32)
        y_class[y == i] = 1
        priv_group, pos_label = 1, 1
        y_pred = _get_preds_for_opportunity_score(edges, i, n_nodes, y)
        scores[i] = equal_opportunity_difference(y_true=pd.Series(y_class), y_pred=y_pred,
                                                 priv_group=priv_group, pos_label=pos_label)
    print(scores)
    return np.mean(np.abs(scores))


def gini(array, eps = 1e-32):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += eps
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

def statistical_parity(edges, y, metric="std"):
    # taken from https://github.com/thunderock/residual2vec_/pull/4
    """
    edges: edge df with source and target column
    y: original labels
    """

    n_nodes = len(y) # number of nodes
    n_edges = edges.shape[0] # number of nodes
    uy, y = np.unique(y, return_inverse=True) # to ensure that the labels are continuous integers starting from zero
    K = len(uy) # number of classes

    # We need two groups at least
    assert K >= 2

    # Group membership matrix, where U[i,k] = 1 if node $i$ belongs to group $k$
    U = sparse.csr_matrix((np.ones_like(y), (np.arange(n_nodes), y)), shape=(n_nodes, K))

    # Number of nodes in each group
    Nk = np.array(U.sum(axis = 0)).reshape(-1)

    # Adjacency matrix
    A = sparse.csr_matrix((np.ones(n_edges), (edges["source"].values, edges["target"].values)), shape=(n_nodes, n_nodes))

    # Make sure that the adjacency matrix is symemtric
    A = A + A.T
    A.data = A.data *0 + 1

    # Calculate the number of edges that appear in each group
    M = U.T @ A @ U

    # Calculate the edge density
    Mdenom = np.outer(Nk, Nk) - np.diag(Nk)
    P = M / Mdenom
    # Calculate the statistical parity

    probs = P[np.triu_indices(K)]
    if metric == "std":
        parity = np.std(probs)
    elif metric == "gini":
        parity = gini(probs)
    return parity

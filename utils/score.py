# @Filename:    score.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        7/29/22 9:46 PM
import numpy as np
from scipy import sparse



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

    probs = np.array(P[np.triu_indices(K)]).reshape(-1)
    if metric == "std":
        parity = np.std(probs)
    elif metric == "gini":
        parity = gini(probs)
    return parity

# @Filename:    score.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        7/29/22 9:46 PM
import numpy as np
import networkx as nx
import pandas as pd
from scipy.sparse import issparse
from aif360.sklearn.metrics import statistical_parity_difference, equal_opportunity_difference
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


def statistical_parity(edges, y):
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
        priv_group, pos_label = 1,1
        y_pred = _get_preds_for_parity_score(edges, i, n_nodes, y)
        scores[i] = statistical_parity_difference(y_true=pd.Series(y),
                                                  y_pred=y_pred,
                                                  priv_group=priv_group, pos_label=pos_label)
    print(scores)
    return np.mean(np.abs(scores))

## 1) need to check if this implementation of statistical parity is correct
## 2) need to implement a sparse version of statistical parity
## 3) need to implement a sparse version of equal opportunity

## need to then create out vectors for gcn and gat for pokec
## use node2vec with crosswalk to create embeddings for pokec
# -*- coding: utf-8 -*-
# @Author: Ashutosh Tiwari
# @Email: checkashu@gmail.com
# @Date:   2023-02-14 23:08:21
# @Filepath: utils/graph.py
from collections import defaultdict

import numpy as np
from scipy.io import loadmat
from scipy.sparse import issparse
from six import iterkeys
from six.moves import range, zip
from tqdm import tqdm, trange
from utils.utils import check_if_symmetric
import multiprocessing
from joblib import Parallel, delayed


class Graph(defaultdict):
    """Efficient basic implementation of nx `Graph' â€“ Undirected graphs with self loops"""

    def __init__(self):
        super(Graph, self).__init__(list)
        self.edge_weights = None
        self.attr = None
        # self.border_score = None
        self.border_distance = None

    def make_undirected(self):
        # t0 = time()
        # print(list(self))

        # for v in list(self):
        #   for other in self[v]:
        #     if v != other:
        #       # assert False, '{} and {} are not connected'.format(v, # other)
        #       assert v in self[other]
        #       print(self[other])
        #       self[other].append(v)
        #       print(self[other])

        # t1 = time()
        # print('make_directed: added missing edges {}s'.format(t1-t0))
        self.make_consistent()
        return self

    def make_consistent(self):
        for k in iterkeys(self):
            self[k] = list(sorted(set(self[k])))
        self.remove_self_loops()
        return self

    def remove_self_loops(self):
        removed = 0
        for x in self:
            if x in self[x]:
                self[x].remove(x)
                removed += 1
        return self


def load_matfile(file_, variable_name="network", undirected=True):
    mat_varables = loadmat(file_)
    mat_matrix = mat_varables[variable_name]
    return from_numpy(mat_matrix, undirected)


def _ramdomwalk_colorfulness(G, v, l):
    v_color = G.attr[v]
    cur = v
    res = 0
    for i in range(l):
        cur = np.random.choice(G[cur])
        if G.attr[cur] != v_color:
            res += 1
    return res / l


def _node_colorfulness(adj, gm, vs, l=2):
    G = from_numpy(adj, undirected=True)
    G.attr = gm
    result = []
    for v in tqdm(vs, desc='assigning colorfulness'):
        res = 0.001 + np.mean([_ramdomwalk_colorfulness(G, v, l) for _ in range(1000)])
        result.append((v, res))
    return result


def _colorfulness(adj, gm, G, l):
    THREADS = multiprocessing.cpu_count() // 2
    vs = [v for v in G]
    # divide the work into chunks
    vs = np.array_split(vs, THREADS)
    
    inputs = [(adj, gm, v, l) for v in vs]
    print("Starting {} threads for CW".format(THREADS))
    
    map_results = Parallel(n_jobs=THREADS)(delayed(_node_colorfulness)(adj, gm, v, l) for v in tqdm(vs))
    # with multiprocessing.Pool(THREADS) as pool:
    #     map_results = pool.starmap(_node_colorfulness, inputs)
    # print(map_results, len(map_results))
    map_results = [item for sublist in tqdm(map_results, desc='merging results') for item in sublist]
    cfn = {k: v for k, v in map_results}
    return cfn


def set_weights(adj, gm, G, exp_, p_bndry, l):
    """
    l = number of classes
    """
    # assert( (s_method[3] in ['bndry', 'revbndry']) and (s_method[5] == 'exp'))
    cfn = _colorfulness(adj, gm, G, l)
    G.edge_weights = dict()
    for v in tqdm(G, desc='assigning_weights'):
        nei_colors = np.unique([G.attr[u] for u in G[v]])
        if nei_colors.size == 0:
            continue
        w_n = [cfn[u] ** exp_ for u in G[v]]
        if nei_colors.size == 1 and nei_colors[0] == G.attr[v]:
            sm = sum(w_n)
            G.edge_weights[v] = [w / sm for w in w_n]
            continue
        G.edge_weights[v] = [None for _ in w_n]
        for cl in nei_colors:
            ind_cl = [i for i, u in enumerate(G[v]) if G.attr[u] == cl]
            w_n_cl = [w_n[i] for i in ind_cl]
            # Z
            sm_cl = sum(w_n_cl)
            if cl == G.attr[v]:
                # p_bndry = \alpha,
                coef = (1 - p_bndry)
            else:
                if G.attr[v] in nei_colors:
                    coef = p_bndry / (nei_colors.size - 1)
                else:
                    coef = 1 / nei_colors.size
            # if (s_method[3] == 'bndry'):
            for i in ind_cl:
                G.edge_weights[v][i] = coef * w_n[i] / sm_cl
            # else:
            #   for i in ind_cl:
            #     G.edge_weights[v][i] = coef * (1 - (w_n[i] / sm_cl)) / (len(ind_cl) - 1)


def from_networkx(G_input, undirected=True):
    G = Graph()

    for idx, x in enumerate(G_input.nodes()):
        for y in iterkeys(G_input[x]):
            G[x].append(y)

    if undirected:
        G.make_undirected()
    return G


def edge_weights_to_sparse(G, sp_mt, ):
    assert issparse(sp_mt), 'sp_mt is not sparse'
    ret = sp_mt.copy()
    n = len(G)
    for i in trange(n, desc="assigning final weights"):
        # fails on self loop
        ret.data[sp_mt.indptr[i]:sp_mt.indptr[i + 1]] = np.array(G[i])
    return ret


def load_matfile(file_, variable_name="network", undirected=True):
    mat_varables = loadmat(file_)
    mat_matrix = mat_varables[variable_name]
    return from_numpy(mat_matrix, undirected)


def from_numpy(x, undirected=True):
    G = Graph()
    if issparse(x):
        cx = x.tocoo()
        for i, j, v in zip(cx.row, cx.col, cx.data):
            G[i].append(j)
    else:
        raise Exception("Only Sparse metrices and undirected graphs are supported at this time.")
    # check if this works for directed graphs as well
    if undirected:
        G.make_undirected()

    G.make_consistent()
    return G

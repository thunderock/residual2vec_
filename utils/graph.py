from collections import defaultdict

import numpy as np
from scipy.io import loadmat
from scipy.sparse import issparse
from six import iterkeys
from six.moves import range, zip

from utils.utils import check_if_symmetric


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


def _node_colorfulness(G, v, l):
    res = 0.001 + np.mean([_ramdomwalk_colorfulness(G, v, l) for _ in range(1000)])
    return (v, res)


def _colorfulness(G, l):
    # cfn = dict()
    # for i, v in enumerate(G):
    #   print(i, ':')
    #   cfn[v] = _node_colorfulness(G, v)
    # return cfn

    # pool = multiprocessing.Pool(multiprocessing.cpu_count())
    # map_results = pool.starmap(_node_colorfulness, [(G, v) for v in G])
    map_results = [_node_colorfulness(G, v, l) for v in G]
    # pool.close()
    cfn = {k: v for k, v in map_results}
    # print(cfn)
    # asdfkjh
    return cfn


def set_weights(G, exp_, p_bndry, l):
    """
    l = number of classes
    """
    # assert( (s_method[3] in ['bndry', 'revbndry']) and (s_method[5] == 'exp'))
    cfn = _colorfulness(G, l)
    G.edge_weights = dict()
    for v in G:
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
    # assert n == sp_mt.shape[0] and check_if_symmetric(
    #     edge_weights), 'edge_weights is not symmetric, i.e. graph is not undirected'
    for i in range(n):
        ret.data[sp_mt.indptr[i]:sp_mt.indptr[i + 1]] = np.array(G[i])
    return ret


def load_matfile(file_, variable_name="network", undirected=True):
    mat_varables = loadmat(file_)
    mat_matrix = mat_varables[variable_name]
    return from_numpy(mat_matrix, undirected)


def from_numpy(x, undirected=True):
    G = Graph()
    if issparse(x) and check_if_symmetric(x):
        cx = x.tocoo()
        for i, j, v in zip(cx.row, cx.col, cx.data):
            G[i].append(j)
    else:
        raise Exception("Only Dense matrices and undirected graphs are supported at this time not yet supported.")
    # check if this works for directed graphs as well
    if undirected:
        G.make_undirected()

    G.make_consistent()
    return G

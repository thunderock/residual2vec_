import logging
import sys
from io import open
from os import path
from time import time
from glob import glob
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Iterable
import random
from random import shuffle
from itertools import product,permutations
from scipy.io import loadmat
from scipy.sparse import issparse
import numpy as np
import pickle


class Graph(defaultdict):
  """Efficient basic implementation of nx `Graph' â€“ Undirected graphs with self loops"""  
  def __init__(self):
    super(Graph, self).__init__(list)
    self.edge_weights = None
    self.attr = None
    # self.border_score = None
    self.border_distance = None


  def make_undirected(self):
    t0 = time()
    print(list(self))

    for v in list(self):
      for other in self[v]:
        print(v, other)
        if v != other:
          # assert False, '{} and {} are not connected'.format(v, other)
          self[other].append(v)

    t1 = time()
    print('make_directed: added missing edges {}s'.format(t1-t0))

    self.make_consistent()
    return self

  def make_consistent(self):
    t0 = time()
    for k in iterkeys(self):
      self[k] = list(sorted(set(self[k])))
    
    t1 = time()
    print('make_consistent: made consistent in {}s'.format(t1-t0))

    self.remove_self_loops()

    return self

  def remove_self_loops(self):

    removed = 0
    t0 = time()

    for x in self:
      if x in self[x]: 
        self[x].remove(x)
        removed += 1
    
    t1 = time()

    print('remove_self_loops: removed {} loops in {}s'.format(removed, (t1-t0)))
    return self

  def check_self_loops(self):
    for x in self:
      for y in self[x]:
        if x == y:
          return True
    
    return False

  def has_edge(self, v1, v2):
    if v2 in self[v1] or v1 in self[v2]:
      return True
    return False

  def degree(self, nodes=None):
    if isinstance(nodes, Iterable):
      return {v:len(self[v]) for v in nodes}
    else:
      return len(self[nodes])

  def order(self):
    "Returns the number of nodes in the graph"
    return len(self)    

  def number_of_edges(self):
    "Returns the number of nodes in the graph"
    return sum([self.degree(x) for x in self.keys()])/2

  def number_of_nodes(self):
    "Returns the number of nodes in the graph"
    return self.order()



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

# cnt_clrf = 0

def  _node_colorfulness(G, v, l):
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


def set_weights(G, method_):
  if method_.startswith('random_walk'):
    s_method = method_.split('_')
    l = int(s_method[2])
    assert( (s_method[3] in ['bndry', 'revbndry']) and (s_method[5] == 'exp'))
    p_bndry = float(s_method[4])
    exp_ = float(s_method[6])
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
        if (s_method[3] == 'bndry'):
          for i in ind_cl:
            G.edge_weights[v][i] = coef * w_n[i] / sm_cl
        else:
          for i in ind_cl:
            G.edge_weights[v][i] = coef * (1 - (w_n[i] / sm_cl)) / (len(ind_cl) - 1)


def from_networkx(G_input, undirected=True):
    G = Graph()

    for idx, x in enumerate(G_input.nodes()):
        for y in iterkeys(G_input[x]):
            G[x].append(y)

    if undirected:
        G.make_undirected()

    return G


def from_numpy(x, undirected=True):
    G = Graph()

    if issparse(x):
        cx = x.tocoo()
        for i,j,v in zip(cx.row, cx.col, cx.data):
            G[i].append(j)
    else:
      raise Exception("Dense matrices not yet supported.")

    if undirected:
        G.make_undirected()

    G.make_consistent()
    return G

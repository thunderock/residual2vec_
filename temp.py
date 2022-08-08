import numpy as np
from scipy.io import loadmat, savemat
import networkx as nx
from utils import graph
import gc

gc.enable()
data_file = 'data/polbooks.gml'
G = nx.read_gml(data_file)
G = nx.relabel.convert_node_labels_to_integers(G, first_label=0, ordering='default')
nodes = G.nodes(data=True)
labels, group_ids = np.unique([n[1]['value'] for n in nodes], return_inverse=True)
print("is directed: ", G.is_directed())
embs = {}
embs['crosswalk'] = nx.adjacency_matrix(G).asfptype()
savemat('/tmp/embs.mat', embs)
gc.collect()

G = graph.load_matfile('/tmp/embs.mat', variable_name='crosswalk', undirected=True)
print(type(G.edge_weights))
weighted = 'random_walk_3_bndry_0.7_exp_4.0'
G.attr = group_ids
graph.set_weights(G, weighted)
print(type(G.edge_weights))
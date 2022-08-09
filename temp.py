# import numpy as np
# from scipy.io import loadmat, savemat
# import networkx as nx
# from utils import graph
# import gc
#
# gc.enable()
# data_file = 'data/polbooks.gml'
# G = nx.read_gml(data_file)
# G = nx.relabel.convert_node_labels_to_integers(G, first_label=0, # ordering='default')
# nodes = G.nodes(data=True)
# labels, group_ids = np.unique([n[1]['value'] for n in nodes], # return_inverse=True)
# print("is directed: ", G.is_directed())
# embs = {}
# embs['crosswalk'] = nx.adjacency_matrix(G).asfptype()
# savemat('/tmp/embs.mat', embs)
# gc.collect()
#
# G = graph.load_matfile('/tmp/embs.mat', variable_name='crosswalk',#  undirected=True)
# print(type(G.edge_weights))
# weighted = 'random_walk_3_bndry_0.7_exp_4.0'
# G.attr = group_ids
# graph.set_weights(G, exp_=2., p_bndry=0.7, l=3)

import graph_tool.all as gt
import graph_embeddings
from models.crosswalk import Crosswalk
from utils.score import statistical_parity
import faiss
import residual2vec as rv
import numpy as np
import pandas as pd
from scipy import sparse
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
import warnings
warnings.filterwarnings("ignore")

from sklearn.neighbors import kneighbors_graph
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


DATA_FILE = 'data/polbooks.gml'
G = nx.read_gml(DATA_FILE)
G = nx.relabel.convert_node_labels_to_integers(G, first_label=0, ordering='default')

nodes = G.nodes(data=True)
labels, group_ids = np.unique([n[1]['value'] for n in nodes], return_inverse=True)

A = nx.adjacency_matrix(G).asfptype()
deg = np.array(A.sum(axis=1)).reshape(-1)
G = nx.from_scipy_sparse_matrix(A)

models = {}
window_length = 5
num_walks = 10
dim = 128

# models["unbiased"] = graph_embeddings.DeepWalk(window_length=window_length, num_walks=num_walks, restart_prob=0)

# models["degree-unbiased"] = rv.residual2vec_sgd(
#     noise_sampler=rv.ConfigModelNodeSampler(),
#     window_length=window_length,
#     num_walks=num_walks,
#     cuda=True,
#     walk_length=80
# )

# models["group-unbiased"] = rv.residual2vec_sgd(
#     noise_sampler=rv.SBMNodeSampler(
#         group_membership=group_ids, window_length=window_length,
#     ),
#     window_length=window_length,
#     num_walks=num_walks,
#     cuda=True,
#     walk_length=80,
# )

# models["fairwalk"] = graph_embeddings.Fairwalk(window_length=window_length, num_walks=num_walks)
# models["fairwalk-group-unbiased"] = graph_embeddings.Fairwalk(
#     window_length=window_length, num_walks=num_walks, group_membership=group_ids
# )
# models['GCN'] = graph_embeddings.GCN()
# models["gcn-doubleK"] = graph_embeddings.GCN(num_default_features=dim * 2)
# models["graphsage"] = graph_embeddings.GraphSage()
# models["graphsage-doubleK"] = graph_embeddings.GraphSage(num_default_features=dim * 2)
# models["gat"] = graph_embeddings.GAT(layer_sizes=[64, 256])
# models["gat-doubleK"] = graph_embeddings.GAT(num_default_features=dim * 2)

models['crosswalk'] = Crosswalk(group_membership=group_ids, window_length=window_length, num_walks=num_walks)

embs = {}

for k, model in tqdm(models.items()):
    print(model.__class__.__name__)
#     sys.stdout = open(os.devnull, 'w')
    emb = model.fit(A).transform(dim=dim)
#     sys.stdout = sys.__stdout__
    embs[k] = emb

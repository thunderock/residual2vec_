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

#################################################
# import graph_tool.all as gt
# import graph_embeddings
# from models.crosswalk import Crosswalk
# from utils.score import statistical_parity
# import faiss
# import residual2vec as rv
# import numpy as np
# import pandas as pd
# from scipy import sparse
# import seaborn as sns
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import networkx as nx
# import warnings
# warnings.filterwarnings("ignore")
#
# from sklearn.neighbors import kneighbors_graph
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import LabelEncoder
#
#
# DATA_FILE = 'data/polbooks.gml'
# G = nx.read_gml(DATA_FILE)
# G = nx.relabel.convert_node_labels_to_integers(G, first_label=0, ordering='default')
#
# nodes = G.nodes(data=True)
# labels, group_ids = np.unique([n[1]['value'] for n in nodes], return_inverse=True)
#
# A = nx.adjacency_matrix(G).asfptype()
# deg = np.array(A.sum(axis=1)).reshape(-1)
# G = nx.from_scipy_sparse_matrix(A)
#
# models = {}
# window_length = 5
# num_walks = 10
# dim = 128
#
# models["unbiased"] = graph_embeddings.DeepWalk(window_length=window_length, num_walks=num_walks, restart_prob=0)
#
# models["degree-unbiased"] = rv.residual2vec_sgd(
#     noise_sampler=rv.ConfigModelNodeSampler(),
#     window_length=window_length,
#     num_walks=num_walks,
#     cuda=True,
#     walk_length=80
# )
#
# models["group-unbiased"] = rv.residual2vec_sgd(
#     noise_sampler=rv.SBMNodeSampler(
#         group_membership=group_ids, window_length=window_length,
#     ),
#     window_length=window_length,
#     num_walks=num_walks,
#     cuda=True,
#     walk_length=80,
# )
#
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
#
# models['crosswalk'] = Crosswalk(group_membership=group_ids, window_length=window_length, num_walks=num_walks)
#
# embs = {}
#
# for k, model in tqdm(models.items()):
#     print(model.__class__.__name__)
# #     sys.stdout = open(os.devnull, 'w')
#     emb = model.fit(A).transform(dim=dim)
# #     sys.stdout = sys.__stdout__
#     embs[k] = emb
#
#
# def reconstruct_graph(emb, n, m):
#     # choose top m edges to reconstruct the graph
#     S = emb @ emb.T
#     S = np.triu(S, k=1)
#     r, c, v = sparse.find(S)
#     idx = np.argsort(-v)[:m]
#     r, c, v = r[idx], c[idx], v[idx]
#     B = sparse.csr_matrix((v, (r, c)), shape=(n, n))
#     B = B + B.T
#     B.data = B.data * 0 + 1
#     return nx.from_scipy_sparse_matrix(B + B.T)
#
# n_edges = int(A.sum() / 2)
# n_nodes = A.shape[0]
# rgraphs = {}
# for k, emb in embs.items():
#     rgraphs[k] = reconstruct_graph(emb, n_nodes, n_edges)
#
# scores = {}
# for k, graph in rgraphs.items():
#     scores[k] = statistical_parity(graph, group_ids)
#     print("class score: ", k, scores[k])
###########################################################
# import networkx as nx
# import residual2vec as rv
# from torch.utils.data import DataLoader
# from residual2vec.residual2vec_sgd import TripletDataset
#
# data_file = 'data/polbooks.gml'
# G = nx.read_gml(data_file)
# G = nx.relabel.convert_node_labels_to_integers(G, first_label=0, ordering='default')
# A = nx.adjacency_matrix(G).asfptype()
# sampler = rv.ConfigModelNodeSampler()
# sampler.fit(A)
# dataset = TripletDataset(adjmat=A, num_walks=10, window_length=5, noise_sampler=sampler, padding_id=A.shape[0], walk_length=80, p=1, q=1, buffer_size=100000, context_window_type="double")
#
# dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1, pin_memory=True)
#
# for i, data in enumerate(dataloader):
#     print(data)
#     if i == 10:
#         break
###########################################################
# from utils.geometric_datasets import Pokec
# from torch_geometric.loader import NeighborLoader
# from utils.link_prediction import *
#
# data = Pokec().data
# train_loader = NeighborLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_neighbors=[NUM_NEIGHBORS] * 2, input_nodes=data.train_mask)
# test_loader = NeighborLoader(data, batch_size=BATCH_SIZE, shuffle=False, num_neighbors=[NUM_NEIGHBORS] * 2, input_nodes=data.test_mask)
#
# models = [
#     GCNLinkPrediction(in_channels=data.num_features, embedding_size=128, hidden_channels=64, num_layers=3).to(DEVICE),
#     GATLinkPrediction(in_channels=data.num_features, embedding_size=128, hidden_channels=64, num_layers=3).to(DEVICE),
# ]
#
# for model in models:
#     print("model_name: {}, params: {}".format(model.__class__.__name__, model.params))
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#     model.fit(train_loader=train_loader, test_loader=test_loader, optimizer=optimizer, log=True, epochs=3)
###########################################################
from utils.link_prediction import *
from dataset import triplet_dataset
from utils.geometric_datasets import Pokec
from torch_geometric.loader import NeighborLoader

d = triplet_dataset.TripletPokecDataset()
ds = triplet_dataset.NeighborEdgeSampler(d, batch_size=256, shuffle=True, num_workers=1)
batch = next(iter(ds))
a, p, n = batch


# d = Pokec().data
# ds = NeighborLoader(d, batch_size=4, shuffle=True, num_neighbors=[10, 10], )
# batch = next(iter(ds))
# node = batch.x
# edge_list = batch.edge_index


model = GATLinkPrediction(in_channels=d.num_features, embedding_size=128, num_heads=5, num_layers=3, hidden_channels=64, num_embeddings=int(torch.max(d.X)))

x = model.forward_i(a[0], a[1], a[2])
y1 = model.forward_o(p[0], p[1], p[2])
y2 = model.forward_o(n[0], n[1], n[2])
Y1 = model.decode(x, y1)
Y2 = model.decode(x, y2)

print(x.shape, y1.shape, y2.shape, Y1, Y2)

model = GCNLinkPrediction(in_channels=d.num_features, embedding_size=128, hidden_channels=64, num_layers=3, num_embeddings=int(torch.max(d.X)))

x = model.forward_i(a[0], a[1], a[2])
y1 = model.forward_o(p[0], p[1], p[2])
y2 = model.forward_o(n[0], n[1], n[2])
Y1 = model.decode(x, y1)
Y2 = model.decode(x, y2)


print(x.shape, y1.shape, y2.shape, Y1, Y2)




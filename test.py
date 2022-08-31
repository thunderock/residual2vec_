

import residual2vec as rv
import numpy as np
import pandas as pd
from scipy import sparse
import networkx as nx
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from utils.config import *

import warnings
warnings.filterwarnings("ignore")



window_length = 5
num_walks = 10
dim = 128
walk_length = 80
NUM_WORKERS = 8
BS = 1


DATA_FILE = 'data/polbooks.gml'
G = nx.read_gml(DATA_FILE)
G = nx.relabel.convert_node_labels_to_integers(G, first_label=0, ordering='default')

nodes = G.nodes(data=True)
labels, group_ids = np.unique([n[1]['value'] for n in nodes], return_inverse=True)

A = nx.adjacency_matrix(G).asfptype()
deg = np.array(A.sum(axis=1)).reshape(-1)
G = nx.from_scipy_sparse_matrix(A)
models, embs = {}, {}

from utils.link_prediction import *
from dataset import triplet_dataset
k = "degree-unbiased-gcn-outvector"

model = rv.residual2vec_sgd(
    noise_sampler=rv.ConfigModelNodeSampler(),
    window_length=window_length,
    num_walks=num_walks,
    walk_length=walk_length
).fit(A)

adjusted_num_walks = np.ceil(
        num_walks
        * np.maximum(
            1,
            model.batch_size
            * model.miniters
            / (model.n_nodes * num_walks * walk_length),
        )
    ).astype(int)
# d = rv.TripletSimpleDataset(
#         adjmat=model.adjmat,
#         group_ids=group_ids,
#         num_walks=adjusted_num_walks,
#         window_length=model.window_length,
#         noise_sampler=model.sampler,
#         padding_id=model.n_nodes,
#         walk_length=model.walk_length,
#         p=model.p,
#         q=model.q,
#         buffer_size=model.buffer_size,
#         context_window_type=model.context_window_type,
#     )

d = triplet_dataset.TripletPokecDataset()
# dataloader = triplet_dataset.NeighborEdgeSampler(d, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, edge_sample_size=3)
dataloader = triplet_dataset.NeighborEdgeSampler(d, batch_size=BS, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, transforming=True)
models[k] = model
# m = GATLinkPrediction(in_channels=d.num_features, embedding_size=128, num_heads=5, num_layers=3, hidden_channels=64, num_embeddings=d.num_embeddings)

# m = GATLinkPrediction(in_channels=d.num_features, embedding_size=128, num_heads=5, num_layers=5, hidden_channels=64, num_embeddings=d.num_embeddings)
m = GCNLinkPrediction(in_channels=d.num_features, embedding_size=128, hidden_channels=64, num_layers=5, num_embeddings=d.num_embeddings)
m.load_state_dict(torch.load('notebooks/{}'.format(k)))
m.to(DEVICE)
m.eval()
emb = np.zeros((len(d), 128))
with torch.no_grad():
    for idx, batch in enumerate(tqdm(dataloader)):
        # emb[idx, :]= m.forward_i(batch[0]).detach().cpu().numpy()
        emb[idx * BS: (idx + 1) * BS, :] = m.forward_i(batch[0]).detach().cpu().numpy()
embs[k] = emb.copy()

for k, v in embs.items():
    print(k, v.shape)

import pickle as pkl
pkl.dump(embs, open('notebooks/emb_{}.pkl'.format(k), 'wb'))

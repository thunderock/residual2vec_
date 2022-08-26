#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
sys.path.insert(0, '..')
# os.environ["CUDA_VISIBLE_DEVICES"]=""


# In[2]:


import residual2vec as rv
import numpy as np
import pandas as pd
from scipy import sparse
import networkx as nx
from torch.utils.data import DataLoader
import torch

import warnings
warnings.filterwarnings("ignore")


# In[3]:



window_length = 5
num_walks = 10
dim = 128
walk_length = 80
NUM_WORKERS = 4


# In[4]:


DATA_FILE = '../data/polbooks.gml'
G = nx.read_gml(DATA_FILE)
G = nx.relabel.convert_node_labels_to_integers(G, first_label=0, ordering='default')

nodes = G.nodes(data=True)
labels, group_ids = np.unique([n[1]['value'] for n in nodes], return_inverse=True)

A = nx.adjacency_matrix(G).asfptype()
deg = np.array(A.sum(axis=1)).reshape(-1)
G = nx.from_scipy_sparse_matrix(A)
models, embs = {}, {}


# # degree-unbiased

# In[5]:



from residual2vec.word2vec import Word2Vec
k = "degree-unbiased"
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
d = rv.TripletSimpleDataset(
        adjmat=model.adjmat,
        group_ids=group_ids,
        num_walks=adjusted_num_walks,
        window_length=model.window_length,
        noise_sampler=model.sampler,
        padding_id=model.n_nodes,
        walk_length=model.walk_length,
        p=model.p,
        q=model.q,
        buffer_size=model.buffer_size,
        context_window_type=model.context_window_type,
    )
dataloader = DataLoader(
        d,
        batch_size=model.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
models[k] = model
m = Word2Vec(vocab_size=A.shape[0] + 1, embedding_size=dim, padding_idx=A.shape[0])
embs[k] = models[k].transform(model=m, dataloader=dataloader)
torch.save(m.state_dict(), k)


# # group-unbiased

# In[6]:


k = "group-unbiased"
model = rv.residual2vec_sgd(
    noise_sampler=rv.SBMNodeSampler(
        group_membership=group_ids, window_length=window_length,
    ),
    window_length=window_length,
    num_walks=num_walks,
    walk_length=walk_length,
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
d = rv.TripletSimpleDataset(
        adjmat=model.adjmat,
        group_ids=group_ids,
        num_walks=adjusted_num_walks,
        window_length=model.window_length,
        noise_sampler=model.sampler,
        padding_id=model.n_nodes,
        walk_length=model.walk_length,
        p=model.p,
        q=model.q,
        buffer_size=model.buffer_size,
        context_window_type=model.context_window_type,
    )
dataloader = DataLoader(
        d,
        batch_size=model.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
models[k] = model
m = Word2Vec(vocab_size=A.shape[0] + 1, embedding_size=dim, padding_idx=A.shape[0])
embs[k] = models[k].transform(model=m, dataloader=dataloader)
torch.save(m.state_dict(), k)


# In[7]:



from utils.link_prediction import *
from dataset import triplet_dataset
k = "degree-unbiased-small-gat"
model = rv.residual2vec_sgd(
    noise_sampler=rv.ConfigModelNodeSampler(),
    window_length=window_length,
    num_walks=num_walks,
    walk_length=walk_length
).fit(A)

# d = triplet_dataset.TripletPokecDataset()
dataloader = triplet_dataset.NeighborEdgeSampler(d, batch_size=model.batch_size, shuffle=True, num_workers=NUM_WORKERS, edge_sample_size=3)
models[k] = model
m = GATLinkPrediction(in_channels=d.num_features, embedding_size=128, num_heads=5, num_layers=3, hidden_channels=64, num_embeddings=d.num_embeddings)
embs[k] = models[k].transform(model=m, dataloader=dataloader)
torch.save(m.state_dict(), k)


# In[8]:


k = "degree-unbiased-small-gcn"
model = rv.residual2vec_sgd(
    noise_sampler=rv.ConfigModelNodeSampler(),
    window_length=window_length,
    num_walks=num_walks,
    walk_length=walk_length
).fit(A)

# d = triplet_dataset.TripletPokecDataset()
dataloader = triplet_dataset.NeighborEdgeSampler(d, batch_size=model.batch_size, shuffle=True, num_workers=NUM_WORKERS)
models[k] = model
m = GCNLinkPrediction(in_channels=d.num_features, embedding_size=128, hidden_channels=64, num_layers=3, num_embeddings=d.num_embeddings)
embs[k] = models[k].transform(model=m, dataloader=dataloader)


torch.save(m.state_dict(), k)


# In[ ]:


d = triplet_dataset.TripletPokecDataset()


# In[ ]:



from utils.link_prediction import *
from dataset import triplet_dataset
k = "degree-unbiased-gat"
model = rv.residual2vec_sgd(
    noise_sampler=rv.ConfigModelNodeSampler(),
    window_length=window_length,
    num_walks=num_walks,
    walk_length=walk_length
).fit(A)

# d = triplet_dataset.TripletPokecDataset()
dataloader = triplet_dataset.NeighborEdgeSampler(d, batch_size=model.batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
models[k] = model
m = GATLinkPrediction(in_channels=d.num_features, embedding_size=128, num_heads=5, num_layers=5, hidden_channels=64, num_embeddings=d.num_embeddings)
embs[k] = models[k].transform(model=m, dataloader=dataloader)
torch.save(m.state_dict(), k)


# In[ ]:


k = "degree-unbiased-gcn"
model = rv.residual2vec_sgd(
    noise_sampler=rv.ConfigModelNodeSampler(),
    window_length=window_length,
    num_walks=num_walks,
    walk_length=walk_length
).fit(A)

# d = triplet_dataset.TripletPokecDataset()
dataloader = triplet_dataset.NeighborEdgeSampler(d, batch_size=model.batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
models[k] = model
m = GCNLinkPrediction(in_channels=d.num_features, embedding_size=128, hidden_channels=64, num_layers=5, num_embeddings=d.num_embeddings)
embs[k] = models[k].transform(model=m, dataloader=dataloader)


torch.save(m.state_dict(), k)


# In[ ]:


for k,i in embs.items():
    print(k, i.shape)


# In[ ]:


# import pickle as pkl
# pkl.dump(embs, open("/tmp/embs.pkl", "wb"))


# In[ ]:





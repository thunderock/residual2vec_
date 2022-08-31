#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
sys.path.insert(0, '..')
# os.environ["CUDA_VISIBLE_DEVICES"]=""



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






from utils.link_prediction import *
from dataset import triplet_dataset

models, embs = {}, {}



k = "degree-unbiased-gcn"
model = rv.residual2vec_sgd(
    noise_sampler=rv.ConfigModelNodeSampler(),
    window_length=window_length,
    num_walks=num_walks,
    walk_length=walk_length
).fit()

d = triplet_dataset.TripletPokecDataset()
dataloader = triplet_dataset.NeighborEdgeSampler(d, batch_size=model.batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
models[k] = model
m = GCNLinkPrediction(in_channels=d.num_features, embedding_size=128, hidden_channels=64, num_layers=5, num_embeddings=d.num_embeddings)
embs[k] = models[k].transform(model=m, dataloader=dataloader)


torch.save(m.state_dict(), k)


# In[ ]:


for k,i in embs.items():
    print(k, i.shape)




# @Filename:    link_prediction.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        8/01/22 8:17 PM
import numpy as np
import torch
import torch.nn as nn
from utils.config import DROPOUT, EMBEDDING_DIM, DEVICE
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F


class LinkPrediction(nn.Module):

    def __init__(self, num_embeddings, classification=False, embedding_size=EMBEDDING_DIM, dropout=DROPOUT, learn_outvec=True):
        super(LinkPrediction, self).__init__()
        self.dropout = dropout
        self.classification = classification
        self.embedding_size = embedding_size
        self.num_embeddings = num_embeddings
        self.ivectors = nn.Linear(self.num_embeddings, self.embedding_size)
        self.ovectors = nn.Linear(self.num_embeddings, self.embedding_size)
        self.learn_outvec = learn_outvec

    @property
    def params(self): return sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, self.parameters())])

    @staticmethod
    def decode(x, y):
        return torch.dot(x.mean(dim=0), y.mean(dim=0))

    def forward(self, node, X, edge_index):
        return self.forward_i((node, X, edge_index))

    def _forward_common(self, X, edge_index):
        # print("edge_index", X.shape)
        X = F.elu(self.in_layer(X, edge_index))
        for idx in range(len(self.layers)):
            X = F.relu(self.layers[idx](X, edge_index))
        X = F.dropout(X, self.dropout, training=self.training)
        return F.relu(self.out_layer(X, edge_index))

    def forward_o(self, X):
        if not self.learn_outvec:
            return self.forward_i(X)
        node, X, edge_index = X[0].to(DEVICE), X[1].to(DEVICE), X[2].to(DEVICE)
        x = self.ovectors(node)
        batch_size = x.size(0)
        y = self._forward_common(X, edge_index)
        X = self.o_lin(torch.cat([x.view(x.size(0), -1).repeat(1, self.repeat_val), y.mean(dim=0).expand(batch_size, -1)], dim=1))
        return self.o_lin_out(X)

    def forward_i(self, X):
        # X = X.to(torch.int64)
        node, X, edge_index = X[0].to(DEVICE), X[1].to(DEVICE), X[2].to(DEVICE)
        x = self.ivectors(node)
        batch_size = x.size(0)
        y = self._forward_common(X, edge_index)
        X = self.i_lin(torch.cat([x.view(x.size(0), -1).repeat(1, self.repeat_val), y.mean(dim=0).expand(batch_size, -1)], dim=1))
        # insert relu here
        X = F.relu(X)
        return self.i_lin_out(X)


class GATLinkPrediction(LinkPrediction):
    def __init__(self, in_channels, hidden_channels, num_layers=2, num_heads=2, **kwargs):
        assert num_layers >= 2 and num_heads >= 1
        super(GATLinkPrediction, self).__init__(**kwargs)
        self.in_layer = GATConv(in_channels=in_channels, out_channels=hidden_channels, heads=num_heads,)
        self.layers = [GATConv(in_channels=hidden_channels * num_heads, out_channels=hidden_channels, heads=num_heads, ) for _ in range(num_layers - 2)]

        self.out_layer = GATConv(in_channels=hidden_channels * num_heads, out_channels=self.embedding_size, heads=num_heads, )

        self.i_lin = torch.nn.Linear(self.embedding_size * num_heads * 2, self.embedding_size, )
        self.o_lin = torch.nn.Linear(self.embedding_size * num_heads * 2, self.embedding_size, )
        self.i_lin_out = torch.nn.Linear(self.embedding_size, self.embedding_size, )
        self.o_lin_out = torch.nn.Linear(self.embedding_size, self.embedding_size, )
        for idx, att in enumerate(self.layers):
            self.add_module('att_{}'.format(idx), att)
        self.repeat_val = 2


class GCNLinkPrediction(LinkPrediction):
    def __init__(self, in_channels, hidden_channels, num_layers=2, num_embeddings=16, **kwargs):
        assert num_layers >= 2
        super(GCNLinkPrediction, self).__init__(num_embeddings=num_embeddings, **kwargs)
        self.in_layer = GCNConv(in_channels=in_channels, out_channels=hidden_channels)
        self.layers = [GCNConv(in_channels=hidden_channels * 1, out_channels=hidden_channels,) for _ in range(num_layers - 2)]
        self.out_layer = GCNConv(in_channels=hidden_channels * 1, out_channels=self.embedding_size * in_channels)
        for idx, att in enumerate(self.layers):
            self.add_module('cnn_{}'.format(idx), att)
        self.i_lin = torch.nn.Linear(self.embedding_size * in_channels * 2, self.embedding_size, )
        self.i_lin_out = torch.nn.Linear(self.embedding_size, self.embedding_size, )
        self.o_lin = torch.nn.Linear(self.embedding_size * in_channels * 2, self.embedding_size, )
        self.o_lin_out = torch.nn.Linear(self.embedding_size, self.embedding_size, )
        self.repeat_val = num_embeddings

# @Filename:    model_utils.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        7/31/22 3:07 PM

import torch
import math
from utils.config import *
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(Module):
    def __init__(self, in_features, out_features, dropout=DROPOUT, alpha=ALPHA, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.dropout = dropout
        self.concat = concat
        self.W = Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = Parameter(torch.zeros(2 * out_features, 1))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def _attention(self, Wh):
        W1 = torch.matmul(Wh, self.a[:self.out_features, :])
        W2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = W1 + W2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ' -> ' \
                + str(self.concat) + ')'

    def forward(self, X, adj):
        W = torch.mm(X, self.W)
        e = self._attention(W)
        zero_vec = -9e15 * torch.ones_like(e)
        att = torch.where(adj > 0, e, zero_vec)
        att = F.softmax(att, dim=1)
        att = F.dropout(att, self.dropout, training=self.training)
        h_prime = torch.matmul(att, W)
        if self.concat:
            return F.elu(h_prime)
        return h_prime


class GraphConvolutionLayer(Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
        # create Weight and Bias trainable parameters
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)

        # N(A) * H * W # Addition aggregation by multiplying
        output = torch.spmm(adj, support)
        return output + self.bias
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

class Aggregator(nn.Module):

    def __init__(self, input_dim=None, output_dim=None, device='cpu'):
        """
        Parameters
        ----------
        input_dim : int or None.
            Dimension of input node features. Used for defining fully
            connected layer in pooling aggregators. Default: None.
        output_dim : int or None
            Dimension of output node features. Used for defining fully
            connected layer in pooling aggregators. Currently only works when
            input_dim = output_dim. Default: None.
        """
        # super(Aggregator, self).__init__()
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

    def forward(self, features, nodes, mapping, rows, num_samples=25):
        """
        Parameters
        ----------
        features : torch.Tensor
            An (n' x input_dim) tensor of input node features.
        nodes : numpy array
            nodes is a numpy array of nodes in the current layer of the computation graph.
        mapping : dict
            mapping is a dictionary mapping node v (labelled 0 to |V|-1) to
            its position in the layer of nodes in the computationn graph
            before nodes. For example, if the layer before nodes is [2,5],
            then mapping[2] = 0 and mapping[5] = 1.
        rows : numpy array
            rows[i] is an array of neighbors of node i which is present in nodes.
        num_samples : int
            Number of neighbors to sample while aggregating. Default: 25.
        Returns
        -------
        out : torch.Tensor
            An (len(nodes) x output_dim) tensor of output node features.
            Currently only works when output_dim = input_dim.
        """
        _choice, _len, _min = np.random.choice, len, min
        mapped_rows = [np.array([mapping[v] for v in row], dtype=np.int64) for row in rows]
        if num_samples == -1:
            sampled_rows = mapped_rows
        else:
            sampled_rows = [_choice(row, _min(_len(row), num_samples), _len(row) < num_samples) for row in mapped_rows]

        n = _len(nodes)
        if self.__class__.__name__ == 'LSTMAggregator':
            out = torch.zeros(n, 2*self.output_dim).to(self.device)
        else:
            out = torch.zeros(n, self.output_dim).to(self.device)
        for i in range(n):
            if _len(sampled_rows[i]) != 0:
                out[i, :] = self._aggregate(features[sampled_rows[i], :])

        return out

    def _aggregate(self, features):
        """
        Parameters
        ----------
        Returns
        -------
        """
        raise NotImplementedError

class MeanAggregator(Aggregator):

    def _aggregate(self, features):
        """
        Parameters
        ----------
        features : torch.Tensor
            Input features.
        Returns
        -------
        Aggregated feature.
        """
        return torch.mean(features, dim=0)

class PoolAggregator(Aggregator):

    def __init__(self, input_dim, output_dim, device='cpu'):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input node features. Used for defining fully connected layer.
        output_dim : int
            Dimension of output node features. Used for defining fully connected layer. Currently only works when output_dim = input_dim.
        """
        # super(PoolAggregator, self).__init__(input_dim, output_dim, device)
        super().__init__(input_dim, output_dim, device)

        self.fc1 = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def _aggregate(self, features):
        """
        Parameters
        ----------
        features : torch.Tensor
            Input features.
        Returns
        -------
        Aggregated feature.
        """
        out = self.relu(self.fc1(features))
        return self._pool_fn(out)

    def _pool_fn(self, features):
        """
        Parameters
        ----------
        Returns
        -------
        """
        raise NotImplementedError

class MaxPoolAggregator(PoolAggregator):

    def _pool_fn(self, features):
        """
        Parameters
        ----------
        features : torch.Tensor
            Input features.
        Returns
        -------
        Aggregated feature.
        """
        return torch.max(features, dim=0)[0]

class MeanPoolAggregator(PoolAggregator):

    def _pool_fn(self, features):
        """
        Parameters
        ----------
        features : torch.Tensor
            Input features.
        Returns
        -------
        Aggregated feature.
        """
        return torch.mean(features, dim=0)[0]

class LSTMAggregator(Aggregator):

    def __init__(self, input_dim, output_dim, device='cpu'):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input node features. Used for defining LSTM layer.
        output_dim : int
            Dimension of output node features. Used for defining LSTM layer. Currently only works when output_dim = input_dim.
        """
        # super(LSTMAggregator, self).__init__(input_dim, output_dim, device)
        super().__init__(input_dim, output_dim, device)

        self.lstm = nn.LSTM(input_dim, output_dim, bidirectional=True, batch_first=True)

    def _aggregate(self, features):
        """
        Parameters
        ----------
        features : torch.Tensor
            Input features.
        Returns
        -------
        Aggregated feature.
        """
        perm = np.random.permutation(np.arange(features.shape[0]))
        features = features[perm, :]
        features = features.unsqueeze(0)

        out, _ = self.lstm(features)
        out = out.squeeze(0)
        out = torch.sum(out, dim=0)

        return out

class GraphAttention(nn.Module):

    def __init__(self, input_dim, output_dim, num_heads, dropout=0.5):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input node features.
        output_dim : int
            Dimension of output features after each attention head.
        num_heads : int
            Number of attention heads.
        dropout : float
            Dropout rate. Default: 0.5.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads

        self.fcs = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_heads)])
        self.a = nn.ModuleList([nn.Linear(2*output_dim, 1) for _ in range(num_heads)])

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=0)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, features, nodes, mapping, rows):

        nprime = features.shape[0]
        rows = [np.array([mapping[v] for v in row], dtype=np.int64) for row in rows]
        sum_degs = np.hstack(([0], np.cumsum([len(row) for row in rows])))
        mapped_nodes = [mapping[v] for v in nodes]
        indices = torch.LongTensor([[v, c] for (v, row) in zip(mapped_nodes, rows) for c in row]).t()

        out = []
        for k in range(self.num_heads):
            h = self.fcs[k](features)

            nbr_h = torch.cat(tuple([h[row] for row in rows if len(row) > 0]), dim=0)
            self_h = torch.cat(tuple([h[mapping[nodes[i]]].repeat(len(row), 1) for (i, row) in enumerate(rows) if len(row) > 0]), dim=0)
            attn_h = torch.cat((self_h, nbr_h), dim=1)

            e = self.leakyrelu(self.a[k](attn_h))

            alpha = [self.softmax(e[lo : hi]) for (lo, hi) in zip(sum_degs, sum_degs[1:])]
            alpha = torch.cat(tuple(alpha), dim=0)
            alpha = alpha.squeeze(1)
            alpha = self.dropout(alpha)

            adj = torch.sparse.FloatTensor(indices, alpha, torch.Size([nprime, nprime]))
            out.append(torch.sparse.mm(adj, h)[mapped_nodes])

        return out
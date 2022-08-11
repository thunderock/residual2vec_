# @Filename:    node_classification.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        7/30/22 8:17 PM

import torch.nn as nn
import torch.nn.functional as F
from utils.model_utils import GraphConvolutionLayer, GraphAttentionLayer
from utils.utils import CONSTANTS
from utils.config import *
from tqdm import trange
import torch
# model should take embeddings, adj list, train labels (supervised and semi supervised (probably need to decide specifications)), return

# TODO (ashutiwa): This should take a data loader

class NodeClassification(nn.Module):

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.semi_supervised) + ' -> ' + str(self.layer_units) + ')'

    def __init__(self, n_nodes, embedding_size, layer_units, semi_supervised=False, local_clf=None, dropout=DROPOUT, classsification=False):
        assert (semi_supervised is False) == (local_clf is None), "semi_supervised and local_clf must be None or both not None"
        super(NodeClassification, self).__init__()
        self.semi_supervised = semi_supervised
        self.local_clf = local_clf
        self.layer_units = layer_units
        self.dropout = dropout
        self.embedding_size = embedding_size
        self.n_nodes = n_nodes
        self.ivectors = nn.Embedding(n_nodes, self.embedding_size)
        self.ovectors = nn.Embedding(n_nodes, self.embedding_size)
        self.classification = classsification
        self.ivectors.weight = nn.Parameter(
            torch.cat(
                [
                    torch.zeros(1, self.embedding_size),
                    torch.FloatTensor(self.n_nodes, self.embedding_size).uniform_(
                        -0.5 / self.embedding_size, 0.5 / self.embedding_size
                    ),
                ]
            )
        )
        self.ovectors.weight = nn.Parameter(
            torch.cat(
                [
                    torch.zeros(1, self.embedding_size),
                    torch.FloatTensor(self.n_nodes, self.embedding_size).uniform_(
                        -0.5 / self.embedding_size, 0.5 / self.embedding_size
                    ),
                ]
            )
        )
        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True


    def transform(self, X, adj_list): return self.forward(X, adj_list)

    def forward(self, X, adj_list):
        return self.forward_i(X, adj_list)

    def forward_i(self, X, adj_list):
        X = self.forward_common(X, adj_list)
        if self.classification:
            X = self.fc(X)
            return F.log_softmax(X, dim=1)
        return self.ivectors(X)

    def forward_o(self, X, adj_list):
        assert not self.classification, "should not be called when classification is False"
        X = self.forward_common(X, adj_list)
        return self.ovectors(X)

    def fit(self, X, y, adj_list, optimizer, loss=CONSTANTS.NLL_LOSS, log=False, epochs=200, eval_set=None):
        if eval_set:
            X_test, y_test = eval_set
        if self.semi_supervised:
            # there need to be eval set for semi supervised learning for now!
            assert eval_set, "eval_set must be provided for semi supervised learning"
            # https: // github.com / ahmadkhajehnejad / CrossWalk / blob / master / classifier / main.py
            y_pred = self.local_clf.fit(X, y).transform(X_test)
            # merge X_test and X
            X = torch.cat((X, X_test), dim=0)
            y = torch.cat((y, y_pred), dim=0)

        for epoch in trange(epochs):
            self.train()
            optimizer.zero_grad()
            output = self.forward(X, adj_list)
            loss_train = loss(output, y)
            if log:
                print("epoch: ", epoch, "loss_train:", loss_train.item())
            loss_train.backward()
            optimizer.step()
            if eval_set and log and not self.semi_supervised:
                self.eval()
                output_test = self.forward(X_test, adj_list)
                loss_test = loss(output_test, y_test)
                print("loss_test:", loss_test.item())
        return self

        def forward_common(self, X, adj_list):
            raise NotImplementedError("forward_common must be implemented")


class GAT(NodeClassification):

    def __init__(self, n_feat, n_nodes, n_hids, nheads, embedding_size=None, classification=False, semi_supervised=False, local_clf=None, n_classes=None):
        super(GAT, self).__init__(n_nodes=n_nodes, embedding_size=embedding_size, classsification=classification, layer_units=nheads, semi_supervised=semi_supervised, local_clf=local_clf)
        self.att = [GraphAttentionLayer(n_feat, n_hids) for _ in range(nheads)]
        for idx, att in enumerate(self.att):
            self.add_module("att_" + str(idx), att)
        self.out_att = GraphAttentionLayer(n_hids * nheads, n_nodes, concat=False)
        if classification:
            self.fc = nn.Linear(n_nodes, n_classes)

    def forward_common(self, X, adj_list):
        X = F.dropout(X, self.dropout, training=self.training)
        X = torch.cat([att(X, adj_list.to_dense()) for att in self.att], dim=1)
        X = F.dropout(X, self.dropout, training=self.training)
        X = F.elu(self.out_att(X, adj_list.to_dense()))
        return X


class GCN(NodeClassification):

    def __init__(self, n_feat, n_nodes, n_hids, embedding_size=None, n_classes=None, classification=False, semi_supervised=False, local_clf=None):
        super(GCN, self).__init__(n_nodes=n_nodes, embedding_size=embedding_size, classsification=classification, layer_units=[n_feat] + n_hids, semi_supervised=semi_supervised, local_clf=local_clf)
        self.graph_layers = nn.ModuleList(
            [GraphConvolutionLayer(self.layer_units[idx], self.layer_units[idx + 1]) for idx in range(len(self.layer_units) - 1)])
        self.output_layer = GraphConvolutionLayer(self.layer_units[-1], n_nodes)
        if classification:
            self.fc = nn.Linear(n_nodes, n_classes)

    def forward_common(self, x, adj):
        for graph_layer in self.graph_layers:
            x = F.relu(graph_layer(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        return self.output_layer(x, adj)
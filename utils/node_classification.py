# @Filename:    node_classification.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        7/30/22 8:17 PM

import torch.nn as nn
import torch.nn.functional as F
from utils.model_utils import GraphConvolutionLayer, GraphAttentionLayer
from utils.utils import CONSTANTS
from utils.config import *
# model should take embeddings, adj list, train labels (supervised and semi supervised (probably need to decide specifications)), return


class NodeClassification(nn.Module):

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.semi_supervised) + ' -> ' + str(self.layer_units) + ')'

    def __init__(self, layer_units, semi_supervised=False, local_clf=None, dropout=DROPOUT):
        assert (semi_supervised is False) == (local_clf is None), "semi_supervised and local_clf must be None or both not None"
        super(NodeClassification, self).__init__()
        self.semi_supervised = semi_supervised
        self.local_clf = local_clf
        self.layer_units = layer_units
        self.dropout = dropout

    def transform(self, X, adj_list): return self.forward(X, adj_list)

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

        for epoch in range(epochs):
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


class GAT(NodeClassification):

    def __init__(self, n_feat, n_hids, n_classes, nheads, semi_supervised=False, local_clf=None):
        super(GAT, self).__init__(layer_units=nheads, semi_supervised=semi_supervised, local_clf=local_clf)
        self.att = [GraphAttentionLayer(n_feat, n_hids) for _ in range(nheads)]
        for idx, att in enumerate(self.att):
            self.add_module("att_" + str(idx), att)
        self.out_att = GraphAttentionLayer(n_hids * nheads, n_classes, concat=False)

    def forward(self, X, adj_list):
        X = F.dropout(X, self.dropout, training=self.training)
        X = torch.cat([att(X, adj_list.to_dense()) for att in self.att], dim=1)
        X = F.dropout(X, self.dropout, training=self.training)
        X = F.elu(self.out_att(X, adj_list.to_dense()))
        return F.log_softmax(X, dim=1)


class GCN(NodeClassification):

    def __init__(self, n_feat, n_hids, n_classes, semi_supervised=False, local_clf=None):
        super(GCN, self).__init__(layer_units=[n_feat] + n_hids, semi_supervised=semi_supervised, local_clf=local_clf)
        self.graph_layers = nn.ModuleList(
            [GraphConvolutionLayer(self.layer_units[idx], self.layer_units[idx + 1]) for idx in range(len(self.layer_units) - 1)])
        self.output_layer = GraphConvolutionLayer(self.layer_units[-1], n_classes)

    def forward(self, x, adj):
        for graph_layer in self.graph_layers:
            x = F.relu(graph_layer(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.output_layer(x, adj)
        return F.log_softmax(x, dim=1)
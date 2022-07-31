# @Filename:    node_classification.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        7/30/22 8:17 PM

import torch.nn as nn
import torch.nn.functional as F
from model_utils import GraphConvolution
# model should take embeddings, adj list, train labels (supervised and semi supervised (probably need to decide specifications)), return

class GCN(nn.Module):
    '''
     Multiple graph convolutional neural network model
     ...
     Attributes
     ----------
     n_feat: int
         The size of the input feature vector of the graph network
     n_hid: int
         The size of the hidden layer, that is, the size of the output vector of the first layer of the convolutional layer
     n_class: int
         Number of classifier categories
     dropout: float
         dropout rate

     Methods
     -------
     __init__(self, n_feat, n_hid, n_class, dropout)
         Two-layer graph convolutional neural network constructor, defining the dimension of the input feature, the dimension of the hidden layer, the number of classifier categories, and the dropout rate
     forward(self, x, adj)
         Forward propagation function, x is the input feature of the graph network, adj is the adjacency matrix that has been transformed $N(A)$
     '''

    def __init__(self, n_feat, n_hids, n_class, dropout):
        super(GCN, self).__init__()

        # Define the layers of graph convolutional layer

        layers_units = [n_feat] + n_hids

        self.graph_layers = nn.ModuleList(
            [GraphConvolution(layers_units[idx], layers_units[idx + 1]) for idx in range(len(layers_units) - 1)])

        self.output_layer = GraphConvolution(layers_units[-1], n_class)

        self.dropout = dropout

    def forward(self, x, adj):
        for graph_layer in self.graph_layers:
            x = F.relu(graph_layer(x, adj))
            # dropout
            x = F.dropout(x, self.dropout, training=self.training)

        # The output of the final convolutional layer is mapped to the output category dimension
        x = self.output_layer(x, adj)

        # Calculate log softmax
        # https://discuss.pytorch.org/t/logsoftmax-vs-softmax/21386/20
        return F.log_softmax(x, dim=1)
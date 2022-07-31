# @Filename:    model_utils.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        7/31/22 3:07 PM

import torch
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(Module):
    """
     A simple implementation of graph convolution, please refer to the paper https://arxiv.org/abs/1609.02907
     ...
     Attributes
     ----------
     in_features: int
         The size of the image convolution input feature vector, namely $|H^{(l)}|$
     out_features: int
         The size of the image convolution output vector, namely $|H^{(l+1)}|$
     bias: bool
         Whether to use the offset vector, the default is True, that is, the default is to use the offset vector
     weight: Parameter
         Trainable parameters in graph convolution,

     Methods
     -------
     __init__(self, in_features, out_features, bias=True)
         The constructor of the graph convolution, defines the size of the input feature, the size of the output vector, whether to use offset, parameters
     reset_parameters(self)
         Initialize the parameters in the graph convolution
     forward(self, input, adj)
         Forward propagation function, input is the feature input, and adj is the transformed adjacency matrix $N(A)=D^{-1}\tilde{A}$. Completing the calculation logic of forward propagation, $N(A) H^{(l)} W^{(l)}$
     __repr__(self)
         Refactored class name expression
     """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()

        # create Weight and Bias trainable parameters
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # standard weight to be uniform
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # H * W
        support = torch.mm(input, self.weight)

        # N(A) * H * W # Addition aggregation by multiplying
        output = torch.spmm(adj, support)

        if self.bias is not None:
            # N(A) * H * W + b
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
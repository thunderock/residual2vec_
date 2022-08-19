# @Filename:    link_prediction.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        8/01/22 8:17 PM
from tqdm import tqdm
import torch.nn as nn
from torch_geometric.utils import negative_sampling
from utils.config import *
from utils.utils import CONSTANTS
from torch_geometric.nn import GCNConv, GATConv
from sklearn.metrics import f1_score
import torch.nn.functional as F
from layers.layers import GATConvMean, GCNConvMean


class LinkPrediction(nn.Module):

    def __init__(self, prediction_threshold=PREDICTION_THRESHOLD, classification=False, embedding_size=EMBEDDING_DIM, dropout=DROPOUT):
        super(LinkPrediction, self).__init__()
        self.dropout = dropout
        self.prediction_threshold = prediction_threshold
        self.classification = classification
        self.embedding_size = embedding_size

    @property
    def params(self): return sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, self.parameters())])

    @staticmethod
    def decode(x, y):
        # cosine similarity
        # x * 128
        return torch.dot(x, y)

    def forward(self, X, edge_index):
        return self.forward_i(X, edge_index)

    def _forward_common(self, X, edge_index):
        X = F.elu(self.in_layer(X, edge_index))
        for idx in range(len(self.layers)):
            X = F.relu(self.layers[idx](X, edge_index))
        X = F.dropout(X, self.dropout, training=self.training)
        return X

    def forward_o(self, X, edge_index):
        # X = X.to(torch.int64)
        # X = self.ivectors(X)
        X = self._forward_common(X, edge_index)
        X = F.relu(self.ovectors(X, edge_index))
        # X = X.to(torch.long)
        return X

    def forward_i(self, X, edge_index):
        # X = X.to(torch.int64)
        # X = self.ivectors(X)
        X = self._forward_common(X, edge_index)
        X = F.relu(self.ivectors(X, edge_index))
        return X
        # X = X.to(torch.long)
        # return torch.mean(X, dim=0)

    # @torch.no_grad()
    # def transform(self, loader, scorer=f1_score, log=False):
    #     """
    #     returns the predictions for the given loader, callers responsibility to set shuffle=False
    #     """
    #     self.eval()
    #     scores = []
    #     preds = []
    #     threshold = torch.tensor([self.prediction_threshold]).to(DEVICE)
    #     for batch in tqdm(loader, desc='Transforming', leave=True):
    #         batch = batch.to(DEVICE)
    #         z = self.forward(batch.x, batch.edge_index)
    #         out = self.decode(z, batch.edge_index).view(-1).sigmoid()
    #         pred = (out > threshold).float() * 1
    #         if log:
    #             scores.append(scorer(np.ones(batch.edge_index.size(1)), pred.cpu().numpy(), ))
    #         preds.append(pred.cpu().numpy())
    #     if log:
    #         print('Test Average {} score: {}'.format(scorer.__name__, np.mean(scores)))
    #     return np.concatenate(preds)
    #
    # def fit(self, train_loader, optimizer, test_loader=None, loss=CONSTANTS.BCE_LOSS, log=False, epochs=EPOCHS, scorer=f1_score):
    #     for epoch in range(epochs):
    #         self.train()
    #         total_loss = examples = 0
    #         for batch in tqdm(train_loader, desc='Training', leave=True):
    #             batch = batch.to(DEVICE)
    #             optimizer.zero_grad()
    #             batch_size = batch.batch_size
    #             # print("edge idx: ", batch.edge_index.shape)
    #             # print("x shape: ", batch.x)
    #             z = self.forward(batch.x, batch.edge_index)
    #             neg_edge_idx = negative_sampling(edge_index=batch.edge_index, num_nodes=batch.num_nodes, num_neg_samples=None, method='sparse')
    #             edge_label_idx = torch.cat([batch.edge_index, neg_edge_idx], dim=-1, )
    #             edge_label = torch.cat([torch.ones(batch.edge_index.size(1)), torch.zeros(neg_edge_idx.size(1))], dim=0).to(DEVICE)
    #             out = self.decode(z, edge_label_idx).view(-1)
    #             loss_ = loss(out, edge_label)
    #             loss_.backward()
    #             optimizer.step()
    #             total_loss += float(loss_) * batch_size
    #             examples += batch_size
    #         if log:
    #             print('Epoch: {}, Training Loss: {:.4f}'.format(epoch, total_loss / examples))
    #         if test_loader:
    #             self.transform(loader=test_loader, log=log, scorer=scorer)
    #     return self


class GATLinkPrediction(LinkPrediction):
    def __init__(self, in_channels, hidden_channels, num_layers=2, num_heads=2, **kwargs):
        assert num_layers >= 2 and num_heads >= 1
        super(GATLinkPrediction, self).__init__(**kwargs)
        self.in_layer = GATConv(in_channels=in_channels, out_channels=hidden_channels, heads=num_heads,)
        self.layers = [GATConv(in_channels=hidden_channels * num_heads, out_channels=hidden_channels, heads=num_heads, ) for _ in range(num_layers - 2)]
        # see if we need an embedding layer as ivector and ovector
        self.ivectors = GATConvMean(in_channels=hidden_channels * num_heads, out_channels=self.embedding_size // num_heads, heads=num_heads,)
        self.ovectors = GATConvMean(in_channels=hidden_channels * num_heads, out_channels=self.embedding_size // num_heads, heads=num_heads, )
        for idx, att in enumerate(self.layers):
            self.add_module('att_{}'.format(idx), att)


class GCNLinkPrediction(LinkPrediction):
    def __init__(self, in_channels, hidden_channels, num_layers=2, **kwargs):
        assert num_layers >= 2
        super(GCNLinkPrediction, self).__init__(**kwargs)
        self.in_layer = GCNConv(in_channels=in_channels, out_channels=hidden_channels)
        self.layers = [GCNConv(in_channels=hidden_channels * 1, out_channels=hidden_channels,) for _ in range(num_layers - 2)]
        self.ivectors = GCNConvMean(in_channels=hidden_channels * 1, out_channels=self.embedding_size,)
        self.ovectors = GCNConvMean(in_channels=hidden_channels * 1, out_channels=self.embedding_size, )
        for idx, att in enumerate(self.layers):
            self.add_module('cnn_{}'.format(idx), att)

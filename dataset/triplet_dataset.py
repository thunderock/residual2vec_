# @Filename:    triplet_dataset.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        8/13/22 11:16 AM
import numpy as np
import pandas as pd
from utils.config import *
from torch.utils.data import Dataset
from torch_geometric.data import download_url
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.utils import negative_sampling
from torch_geometric.loader.neighbor_sampler import EdgeIndex
from torch_sparse import SparseTensor


class TripletGraphDataset(Dataset):
    def __init__(self, X: torch.Tensor, edge_index: torch.Tensor, sampler=negative_sampling):
        super().__init__()
        self.X = X
        self.edge_index = edge_index
        self.num_features = self.X.shape[1]

        self.neg_edge_index = sampler(edge_index=self.edge_index, num_nodes=self.X.shape[0], num_neg_samples=None, method='sparse', force_undirected=True)
        self.num_embeddings = int(torch.max(self.X).item()) + 1
        self.n_nodes = self.X.shape[0]

    def __len__(self):
        # assumes that all the nodes ids are present starting from 0 to the max number of nodes
        return self.n_nodes

    def _get_node_edges_from_source(self, idx, edge_index=None, two_dim=False):
        edge_index = edge_index if edge_index is not None else self.edge_index
        mask = edge_index[0, :] == idx
        ret = edge_index[1, mask].squeeze()
        if two_dim:
            return torch.cat([torch.full_like(ret, idx).reshape(-1, 1), ret.reshape(-1, 1)], dim=1).T
        return ret

    def _ret_features_for_node(self, idx):
        # index = self.idx_mapping[idx]self.features
        return idx
        return self.X[idx]

    def _select_random_neighbor(self, source, neg=False):
        edge_index = self.neg_edge_index if neg else self.edge_index
        nodes = self._get_node_edges_from_source(source, edge_index)
        if nodes.dim() == 0 or nodes.shape[0] == 0:
            # this should happen rarely, this means that the node has no edges
            # in that case we randomly sample a node with an edge
            return None
        return nodes[torch.randint(nodes.shape[0], (1,))].squeeze()

    def __getitem__(self, idx):
        """
        returns a, p, n tuple for this idx where a is the current node, p is the positive node and n is the randomly sampled negative node
        """
        # select a node with positive edge
        p_node = self._select_random_neighbor(idx)
        n_node = self._select_random_neighbor(idx, neg=True)
        if p_node and n_node:
            a, p, n = self._ret_features_for_node(idx), self._ret_features_for_node(p_node), self._ret_features_for_node(n_node)
        else:
            idx = torch.randint(self.edge_index.shape[0], (1,))
            # print("randomly selected a_idx", idx)
            # idx = 102
            # select nodes randomly here
            a = self._ret_features_for_node(self.edge_index[0, idx].item())
            # p cannot be none now
            p = self._ret_features_for_node(self._select_random_neighbor(idx))
            # this can still result in failure but haven't seen it yet, this means that negative sampling couldn't generate a negative node for this source node
            n = self._ret_features_for_node(self._select_random_neighbor(idx, neg=True))
        return torch.tensor([a, p, n])


class NeighborEdgeSampler(torch.utils.data.DataLoader):
    def __init__(self, dataset, edge_sample_size=None, transforming=False, **kwargs):
        # investigate dual calling behaviour here, ideal case is calling this class with node_id range dataset
        # node_idx = torch.arange(self.adj_t.sparse_size(0))

        super().__init__(dataset=dataset, collate_fn=self.sample, **kwargs)
        self.features = self.dataset.num_features
        edge_index = dataset.edge_index
        # print(dataset.edge_index.dtype)
        num_nodes = self.dataset.X.shape[0]
        self.features = dataset.num_features
        self.adj_t = self._get_adj_t(edge_index, num_nodes)
        self.neg_adj_t = self._get_adj_t(dataset.neg_edge_index, num_nodes)
        self.neg_adj_t.storage.rowptr()
        self.adj_t.storage.rowptr()
        if not edge_sample_size:
            edge_sample_size = num_nodes // 2
        self.edge_sample_size = torch.tensor(edge_sample_size)
        self.transforming = transforming

    def _get_adj_t(self, edge_index, num_nodes):
        return SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.arange(edge_index.size(1)), sparse_sizes=(num_nodes, num_nodes)).t()

    def sample(self, batch):
        # make sure this is a list of tensors
        if not isinstance(batch[0], torch.Tensor):
            batch = [torch.tensor(b) for b in batch]
        batch = torch.stack(batch)
        a, p, n = batch[:, 0], batch[:, 1], batch[:, 2]
        adjs, nids = [], []
        if self.transforming:
            # idx = 0
            # n_id = a
            adj_t, node_ids = self.adj_t.sample_adj(a, self.edge_sample_size, replace=False)
            # e_id = adj_t.storage.value()
            # size = adj_t.sparse_sizes()[::-1]
            row, col, _ = adj_t.coo()
            edge_index = torch.stack([row, col], dim=0)
            # adjs.append(EdgeIndex(edge_index, e_id, size))
            # adjs = [edge_index] * 3
            # nids = [n_id] * 3
            x = (self.dataset.X[a], self.dataset.X[node_ids], edge_index)
            return x, x, x
        else:
            for idx, n_id in enumerate((a, p, n)):
                if idx == 2:
                    # in case of negativly sampled node
                    adj_t, node_ids = self.neg_adj_t.sample_adj(n_id, self.edge_sample_size, replace=False)
                else:
                    adj_t, node_ids = self.adj_t.sample_adj(n_id, self.edge_sample_size, replace=False)
                # e_id = adj_t.storage.value()
                # size = adj_t.sparse_sizes()[::-1]
                row, col, _ = adj_t.coo()
                edge_index = torch.stack([row, col], dim=0)
                # adjs.append(EdgeIndex(edge_index, e_id, size))
                adjs.append(edge_index)
                nids.append(node_ids)
        # get this features from dataset itself in future
        return (self.dataset.X[a], self.dataset.X[nids[0]], adjs[0]), \
               (self.dataset.X[p], self.dataset.X[nids[1]], adjs[1]), \
               (self.dataset.X[n], self.dataset.X[nids[2]], adjs[2])

    def __repr__(self) -> str:
        return '{}({}, batch_size={})'.format(self.__class__.__name__, len(self.dataset), self.batch_size)



class SbmSamplerWrapper(object):
    def __init__(self, adj_path, group_membership, window_length, num_edges, use_weights=True, **params):
        from residual2vec.node_samplers import SBMNodeSampler
        from residual2vec.residual2vec_sgd import TripletSimpleDataset
        from scipy import sparse
        from residual2vec import utils
        sampler = SBMNodeSampler(window_length=window_length, group_membership=group_membership, dcsbm=True)
        adj = sparse.load_npz(adj_path)
        if not use_weights:
            adj.data = np.ones_like(adj.data)
        n_nodes = adj.shape[0]
        adj = utils.to_adjacency_matrix(adj)
        sampler.fit(adj)
        dataset = TripletSimpleDataset(adjmat=adj, group_ids=group_membership, noise_sampler=sampler, **params, buffer_size=n_nodes, window_length=window_length)
        self.num_edges = num_edges
        centers, contexts, random_contexts = dataset.centers, dataset.contexts, dataset.random_contexts
        indices = np.random.choice(len(centers), num_edges, replace=False)
        self.centers, contexts, random_contexts = centers[indices], contexts[indices], random_contexts[indices]
        # be careful, cant call this again, contexts lost
        self.edge_index = self._create_edge_index(self.centers, contexts)


    def _create_edge_index(self, source: np.ndarray, dist: np.ndarray):
        return torch.tensor([source, dist], dtype=torch.long)

    # def sample_neg_edges(self, edge_index, num_nodes, num_neg_samples, method,
    #         force_undirected):
    #     # none of these params used, only for compatibility
    #     return self._create_edge_index(self.centers, self.random_contexts)


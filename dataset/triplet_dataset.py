# @Filename:    triplet_dataset.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        8/13/22 11:16 AM
import numpy as np
from torch.utils.data import Dataset
import torch
from torch_sparse import SparseTensor


class TripletGraphDataset(Dataset):
    def __init__(self, X: torch.Tensor, edge_index: torch.Tensor, sampler, num_neg_sampling=3):
        super().__init__()
        self.X = X
        self.edge_index = edge_index
        self.num_features = self.X.shape[1]

        # self.neg_edge_index = sampler(edge_index=self.edge_index, num_nodes=self.X.shape[0], num_neg_samples=None, method='sparse', force_undirected=True)
        self.neg_edge_index = torch.unique(torch.cat([sampler(edge_index=self.edge_index, num_nodes=self.X.shape[0],num_neg_samples=None, method='sparse', force_undirected=True) for _ in range(num_neg_sampling)],dim=1), dim=1)
        print("size of edge_index", self.edge_index.shape, " and size of neg_edge_index", self.neg_edge_index.shape)
        self.n_nodes = self.X.shape[0]
        # sources which are common in both positive and negative edge index
        edge_idx_sources, neg_edge_idx_sources = self.edge_index[0, :], self.neg_edge_index[0, :]
        common_sources = torch.unique(edge_idx_sources[torch.isin(edge_idx_sources, neg_edge_idx_sources)])
        print("number of common sources", common_sources.shape[0])
        assert common_sources.shape[0] == self.n_nodes, "all the nodes should be present in the both positive and negative edge index"

    def __len__(self):
        # assumes that all the nodes ids are present starting from 0 to the max number of nodes
        return self.n_nodes

    def _get_node_edges_from_source(self, idx, edge_index=None):
        edge_index = edge_index if edge_index is not None else self.edge_index
        mask = edge_index[0, :] == idx
        ret = edge_index[1, mask].squeeze()
        if ret.dim() == 0:
            ret = ret.unsqueeze(0)
        elif ret.shape[0] == 0:
            ret = None
        return ret


    def _select_random_neighbor(self, source, neg=False):
        edge_index = self.neg_edge_index if neg else self.edge_index
        nodes = self._get_node_edges_from_source(source, edge_index)
        if not isinstance(nodes, torch.Tensor):
            return None
        return nodes[torch.randint(nodes.shape[0], (1,))].squeeze()

    def __getitem__(self, a):
        """
        returns a, p, n tuple for this idx where a is the current node, p is the positive node and n is the randomly sampled negative node
        """
        # select a node with positive edge
        p = self._select_random_neighbor(a)
        n = self._select_random_neighbor(a, neg=True)

        return torch.tensor([a, p, n])


class NeighborEdgeSampler(torch.utils.data.DataLoader):
    def __init__(self, dataset, edge_sample_size=None, transforming=False, **kwargs):


        super().__init__(dataset=dataset, collate_fn=self.sample, **kwargs)
        self.features = self.dataset.num_features
        edge_index = dataset.edge_index
        # print(dataset.edge_index.dtype)
        num_nodes = self.dataset.X.shape[0]
        self.features = dataset.num_features
        self.adj_t = self._get_adj_t(edge_index, num_nodes)
        self.adj_t.storage.rowptr()
        if not edge_sample_size:
            edge_sample_size = num_nodes
        self.edge_sample_size = torch.tensor(edge_sample_size)
        self.transforming = transforming

    def _get_adj_t(self, edge_index, num_nodes):
        return SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.ones(edge_index.size(1)), sparse_sizes=(num_nodes, num_nodes)).t()

    def sample(self, batch):
        # make sure this is a list of tensors
        if not isinstance(batch[0], torch.Tensor):
            batch = [torch.tensor(b) for b in batch]
        batch = torch.stack(batch)
        a, p, n = batch[:, 0], batch[:, 1], batch[:, 2]

        x = 3 * [None]
        if self.transforming:
            adj_t, node_ids = self.adj_t.sample_adj(a, self.edge_sample_size, replace=False)
            row, col, _ = adj_t.coo()
            edge_index = torch.stack([row, col], dim=0)
            x = (self.dataset.X[a], self.dataset.X[node_ids], edge_index)
            return x, x, x
        else:
            for idx, n_id in enumerate((a, p, n)):
                adj_t, node_ids = self.adj_t.sample_adj(n_id, self.edge_sample_size, replace=False)
                row, col, _ = adj_t.coo()
                edge_index = torch.stack([row, col], dim=0)
                x[idx] = (self.dataset.X[n_id], self.dataset.X[node_ids], edge_index)
        # get this features from dataset itself in future
        return x

    def __repr__(self) -> str:
        return '{}({}, batch_size={})'.format(self.__class__.__name__, len(self.dataset), self.batch_size)



class SbmSamplerWrapper(object):
    def __init__(self, adj_path, group_membership, window_length, num_edges, padding_id, num_walks, use_weights=True, dcsbm=True):
        from residual2vec.node_samplers import SBMNodeSampler

        from scipy import sparse
        from residual2vec import utils
        self.sampler = SBMNodeSampler(window_length=window_length, group_membership=group_membership, dcsbm=dcsbm)
        adj = adj_path if sparse.issparse(adj_path) else sparse.load_npz(adj_path)
        if not use_weights:
            adj.data = np.ones_like(adj.data)
        self.n_nodes = adj.shape[0]
        self.adj = utils.to_adjacency_matrix(adj)
        self.sampler.fit(adj)
        self.num_edges = num_edges
        self.window_length = window_length
        self.padding_id = padding_id
        self.num_walks = num_walks


    def _create_edge_index(self, source: np.ndarray, dist: np.ndarray):
        return torch.tensor([source, dist], dtype=torch.long)

    def sample_neg_edges(self, edge_index, num_nodes, num_neg_samples, method,
            force_undirected):
        from residual2vec.residual2vec_sgd import TripletSimpleDataset
        dataset = TripletSimpleDataset(adjmat=self.adj, noise_sampler=self.sampler, buffer_size=self.n_nodes,
                                       window_length=self.window_length, padding_id=self.padding_id, num_walks=self.num_walks)
        centers, _, random_contexts = dataset.centers, dataset.contexts, dataset.random_contexts
        # none of these params used, only for compatibility
        return self._create_edge_index(centers, random_contexts)



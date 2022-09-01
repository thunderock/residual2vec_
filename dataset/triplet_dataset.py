# @Filename:    triplet_dataset.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        8/13/22 11:16 AM
import pandas as pd
from utils.config import *
from torch.utils.data import Dataset
from torch_geometric.data import download_url
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.utils import negative_sampling
from torch_geometric.loader.neighbor_sampler import EdgeIndex
from torch_sparse import SparseTensor
from torch_geometric.nn.models import Node2Vec
from tqdm import tqdm, trange
from dataset import pokec_data

class TripletPokecDataset(Dataset):
    def __init__(self, root: str = '/tmp/', node2vec: Node2Vec = None,):
        if node2vec:
            # train node2vec here
            loader = node2vec.loader(batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
            optimizer = torch.optim.Adam(list(node2vec.parameters()), lr=0.01)
            self.X = self._train_node2vec(node2vec, loader, optimizer)
            self.edge_index = node2vec.edge_index
        else:
            pokec = pokec_data.PokecDataFrame(root=root)
            self.X = pokec.X
            self.edge_index = pokec.edge_index
        self.num_features = self.X.shape[1]

        self.neg_edge_index = negative_sampling(edge_index=self.edge_index, num_nodes=self.X.shape[0], num_neg_samples=None, method='sparse', force_undirected=True)
        self.num_embeddings = int(torch.max(self.X).item()) + 1
        self.n_nodes = self.X.shape[0]

    def __len__(self):
        # assumes that all the nodes ids are present starting from 0 to the max number of nodes
        return self.n_nodes

    @staticmethod
    def _train_node2vec(model, loader, optimizer, epochs=EPOCHS):
        t = trange(epochs)
        print("training node2vec")
        for epoch in t:
            model.train()
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(DEVICE), neg_rw.to(DEVICE))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_loss /= len(loader)
            t.set_description(f'Epoch {epoch + 1:03d}, Loss: {total_loss:.4f}')
            t.refresh()
        return model.embedding.weight

    def _get_node_edges_from_source(self, idx, edge_index=None, two_dim=False):
        edge_index = edge_index if edge_index is not None else self.edge_index
        mask = edge_index[0, :] == idx
        ret = edge_index[1, mask].squeeze()
        if two_dim:
            return torch.cat([torch.full_like(ret, idx).reshape(-1, 1), ret.reshape(-1, 1)], dim=1).T
        return ret

    def get_adjacency_mt(self):
        pass

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
            x = (self.dataset.X[a].to(torch.long), self.dataset.X[node_ids], edge_index)
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
        return (self.dataset.X[a].to(torch.long), self.dataset.X[nids[0]], adjs[0]), \
               (self.dataset.X[p].to(torch.long), self.dataset.X[nids[1]], adjs[1]), \
               (self.dataset.X[n].to(torch.long), self.dataset.X[nids[2]], adjs[2])

    def __repr__(self) -> str:
        return '{}({}, batch_size={})'.format(self.__class__.__name__, len(self.dataset), self.batch_size)

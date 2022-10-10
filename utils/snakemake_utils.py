import os
from os.path import join as j

import numpy as np
import torch
from models import weighted_node2vec
from utils.config import DEVICE
from scipy import sparse

def get_string_boolean(string):
    if string in ['True', 'true', 'TRUE', 'T', 't', '1']:
        return True
    elif string in ['False', 'false', 'FALSE', 'F', 'f', '0']:
        return False
    else:
        raise ValueError('String must be either True or False')


class FileResources(object):
    def __init__(self, root: str, crosswalk: bool, baseline: bool, model_name:str, basename: str='pokec'):
        self.root = root
        self.crosswalk = crosswalk
        self.baseline = baseline
        self.basename = basename
        assert model_name in ['gat', 'gcn']
        self.model_name = model_name

    @property
    def adj_path(self):
        if self.crosswalk:
            return str(j(self.root, f'{self.basename}_adj_crosswalk.npz'))
        else:
            return str(j(self.root, f'{self.basename}_adj.npz'))

    @property
    def test_adj_path(self): 
        if self.crosswalk:
            return str(j(self.root, "{}_crosswalk_test_adj.npz".format(self.basename)))
        else:
            return str(j(self.root, "{}_test_adj.npz".format(self.basename)))
    @property
    def node2vec_embs(self):
        if self.crosswalk:
            return str(j(self.root, "{}_crosswalk_node2vec.npy".format(self.basename)))
        else:
            return str(j(self.root, "{}_node2vec.npy".format(self.basename)))

    @property
    def model_weights(self):
        if self.baseline:
            if self.crosswalk:
                return str(j(self.root, "{}_crosswalk_{}.h5".format(self.basename, self.model_name)))
            else:
                return str(j(self.root, "{}_{}.h5".format(self.basename, self.model_name)))
        else:
            if self.crosswalk:
                return str(j(self.root, "{}_crosswalk_{}_r2v.h5".format(self.basename, self.model_name)))
            else:
                return str(j(self.root, "{}_{}_r2v.h5".format(self.basename, self.model_name)))

    @property
    def embs_file(self):
        if self.baseline:
            if self.crosswalk:
                return str(j(self.root, "{}_crosswalk_{}_node2vec_embs.npy".format(self.basename, self.model_name)))
            else:
                return str(j(self.root, "{}_{}_node2vec_embs.npy".format(self.basename, self.model_name)))
        else:
            if self.crosswalk:
                return str(j(self.root, "{}_crosswalk_{}_r2v_node2vec_embs.npy".format(self.basename, self.model_name)))
            else:
                return str(j(self.root, "{}_{}_r2v_node2vec_embs.npy".format(self.basename, self.model_name)))


def get_dataset(name):
    dataset = None
    if name == 'pokec':
        from dataset.pokec_data import PokecDataFrame
        dataset = PokecDataFrame()
    elif name == 'small_pokec':
        from dataset.pokec_data import SmallPokecDataFrame
        dataset = SmallPokecDataFrame()
    elif name == 'airport':
        from dataset.airportnet_data import AirportNetDataFrame
        dataset = AirportNetDataFrame()
    elif name == 'polbook':
        from dataset.polbook_data import PolBookDataFrame
        dataset = PolBookDataFrame()
    elif name == 'polblog':
        from dataset.polblog_data import PolBlogDataFrame
        dataset = PolBlogDataFrame()
    # add other datasets here
    return dataset

def _get_node2vec_model(crosswalk, embedding_dim, num_nodes, edge_index, weighted_adj_path=None, group_membership=None):
    if crosswalk:
        # assert weighted_adj_path is not None and group_membership is not None
        return weighted_node2vec.WeightedNode2Vec(
            num_nodes=num_nodes,
            group_membership=group_membership,
            weighted_adj=weighted_adj_path,
            edge_index=edge_index,
            embedding_dim=embedding_dim,
        )
    return weighted_node2vec.UnWeightedNode2Vec(
            num_nodes=num_nodes,
            embedding_dim=embedding_dim,
            weighted_adj=weighted_adj_path,
            edge_index=edge_index
        )


def get_node2vec_trained_get_embs(file_path):
    return torch.from_numpy(np.load(file_path).astype(np.float32))


def train_node2vec_get_embs(file_path, **kwargs):
    model = _get_node2vec_model(**kwargs)
    return model.train_and_get_embs(file_path)

def store_crosswalk_weights(file_path, edge_index, **kwargs):
    # make this edge index symmetric
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    model = _get_node2vec_model(edge_index=edge_index, **kwargs)
    sparse.save_npz(file_path, model.adj)

def get_num_nodes_from_adj(adj_path):
    return sparse.load_npz(adj_path).shape[0]

def get_edge_index_from_sparse_path(weighted_adj):
    adj = sparse.load_npz(weighted_adj)
    row, col = adj.nonzero()
    return torch.cat((torch.from_numpy(row).unsqueeze(dim=0), torch.from_numpy(col).unsqueeze(dim=0))).long()


def return_new_graph(node_embeddings, n_neighbors, batch_size=2000):
    from utils.graph_utils import get_edges_fastknn_faiss
    edges = get_edges_fastknn_faiss(node_embeddings, n_neighbors, batch_size=batch_size)
    # drop rows with target = -1
    return edges.drop(edges[edges['target'] == -1].index)

def get_torch_sparse_from_edge_index(edge_index, num_nodes):
    row, col = edge_index
    from torch_sparse import SparseTensor
    return SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes)).to('cpu')



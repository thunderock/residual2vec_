# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-01-18 00:55:24
# @Last Modified by:   Ashutosh Tiwari
# @Last Modified time: 2023-05-01 18:54:52
from os.path import join as j

import numpy as np
import torch
from models import weighted_node2vec
from utils.config import R2V_TRAINING_EPOCHS, NUM_GNN_LAYERS, DATASET_BATCH_SIZE, DATASET_LEARNING_RATE, NUM_THREADS
from scipy import sparse
import pandas as pd
from snakemake.utils import Paramspace
import itertools
from tqdm import tqdm, trange
import networkx as nx

def get_string_boolean(string):
    # print(string, type(string))
    if isinstance(string, bool):
        return string
    elif string in ['True', 'true', 'TRUE', 'T', 't', '1']:
        return True
    elif string in ['False', 'false', 'FALSE', 'F', 'f', '0']:
        return False
    else:
        raise ValueError('String must be either True or False')


class FileResources(object):
    def __init__(self, root: str, fairwalk:bool, crosswalk: bool, r2v: bool, model_name:str, node2vec:bool, dataset:str, degree_agnostic:bool):
        self.root = root
        self.crosswalk = crosswalk
        self.r2v = r2v
        self.fairwalk = fairwalk
        self.model_name = model_name
        self.node2vec = node2vec
        self.dataset = dataset
        self.degree_agnostic = degree_agnostic

    @property
    def adj_path(self):
        if self.crosswalk:
            return str(j(self.root, f'{self.dataset}_adj_crosswalk.npz'))
        elif self.fairwalk:
            return str(j(self.root, f'{self.dataset}_adj_fairwalk.npz'))
        else:
            return str(j(self.root, f'{self.dataset}_adj.npz'))

    @property
    def test_adj_path(self):
        return str(j(self.root, "{}_test_adj.npz".format(self.dataset)))

    @property
    def baseline_embs_path(self):
        if self.node2vec:
            return str(j(self.root, "{}_baseline_man_woman+node2vec_embs.npy".format(self.dataset)))

        return str(j(self.root, "{}_baseline_man_woman+deepwalk_embs.npy".format(self.dataset)))
    
    @property
    def feature_embs(self):
        if self.crosswalk:
            if self.node2vec:
                return str(j(self.root, "{}_crosswalk_node2vec.npy".format(self.dataset)))
            else:
                return str(j(self.root, "{}_crosswalk_deepwalk.npy".format(self.dataset)))
        elif self.fairwalk:
            # faiwalk
            if self.node2vec:
                return str(j(self.root, "{}_fairwalk_node2vec.npy".format(self.dataset)))
            else:
                return str(j(self.root, "{}_fairwalk_deepwalk.npy".format(self.dataset)))
        else:
            # no walk
            if self.node2vec:
                return str(j(self.root, "{}_node2vec.npy".format(self.dataset)))
            else:
                return str(j(self.root, "{}_deepwalk.npy".format(self.dataset)))

    @property
    def model_weights(self):
        feature_method = "node2vec" if self.node2vec else "deepwalk"
        negative_sampling = "deepwalk"
        if self.r2v: negative_sampling = "r2v"
        if self.model_name == "residual2vec":
            if self.degree_agnostic:
                return str(j(self.root, f"{self.dataset}_{self.model_name}_groupbiased.h5"))
            return str(j(self.root, f"{self.dataset}_{self.model_name}.h5"))
        if self.degree_agnostic:
            return str(j(self.root, f"{self.dataset}_{self.model_name}_{feature_method}_{negative_sampling}_groupbiased.h5"))
        return str(j(self.root, f"{self.dataset}_{self.model_name}_{feature_method}_{negative_sampling}.h5"))

    @property
    def embs_file(self):
        feature_method = "node2vec" if self.node2vec else "deepwalk"
        negative_sampling = "deepwalk"
        if self.r2v: negative_sampling = "r2v"
        if self.model_name == "residual2vec":
            if self.degree_agnostic:
                return str(j(self.root, f"{self.dataset}_{self.model_name}_groupbiased_embs.npy"))
                
            return str(j(self.root, f"{self.dataset}_{self.model_name}_embs.npy"))
        if self.degree_agnostic:
            return str(j(self.root, f"{self.dataset}_{self.model_name}_{feature_method}_{negative_sampling}_groupbiased_embs.npy"))
        return str(j(self.root, f"{self.dataset}_{self.model_name}_{feature_method}_{negative_sampling}_embs.npy"))

def get_dataset(name='generic', **kwargs):
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
    elif name == "facebook":
        from dataset.facebook_data import FacebookData
        dataset = FacebookData()
    elif name == 'copenhagen':
        from dataset import copenhagen_data
        dataset = copenhagen_data.CopenhagenData()
    elif name == 'twitch':
        from dataset import twitch_data
        dataset = twitch_data.TwitchData(group_col='language', **kwargs)
    elif name == 'generic':
        from dataset import generic_data
        assert 'edge_index' in kwargs and 'group_membership' in kwargs, "edge_index and group_membership must be provided for generic dataset"
        dataset = generic_data.GenericData(**kwargs)
    # add other datasets here
    return dataset

def get_adj_from_edge_index(edge_index, num_nodes):
    
    row, col = edge_index
    from torch_sparse import SparseTensor
    adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    row, col, _ = adj.coo()
    ones = np.ones(row.shape[0], dtype=np.int32)
    A = sparse.csr_matrix((ones, (row.numpy(), col.numpy())), shape=(num_nodes, num_nodes))
    return A

def get_networkx_graph(dataset):
    from torch_geometric.utils import to_networkx
    from torch_geometric.data import Data
    data = Data(edge_index=dataset.edge_index, 
            y=dataset.get_grouped_col())
    data.num_nodes = len(dataset.get_grouped_col())

    return to_networkx(data=data, to_undirected=True, node_attrs=['y'])


def get_graph_tool_graph(dataset):
    import graph_tool.all as gt
    g = gt.Graph(directed=False)
    edge_index = dataset.edge_index
    # sort and remove duplicates
    edge_index = edge_index[:, edge_index[0, :] <= edge_index[1, :]]
    g.add_edge_list(edge_index.numpy().T)
    y = g.new_vertex_property("int32_t")
    y.a = dataset.get_grouped_col().numpy()
    g.vertex_properties["y"] = y
    return g

def get_inf_modularity(dataset, **kwargs):
    dataset = get_dataset(dataset, **kwargs)
    g = get_graph_tool_graph(dataset)
    N = dataset.get_grouped_col().shape[0]
    y = g.vertex_properties['y']
    import graph_tool.all as gt
    en_ = gt.BlockState(g, b=y).entropy()
    m = gt.BlockState(g, b=np.zeros(N, dtype=np.int32)).entropy()
    return 1 - (en_ / m)

def _get_deepwalk_model(embedding_dim, num_nodes, weighted_adj_path=None, group_membership=None, crosswalk=True, fairwalk=False):
    assert not (crosswalk and fairwalk), "Both crosswalk and fairwalk cannot be true"
    from models import weighted_deepwalk
    if crosswalk:
        # assert weighted_adj_path is not None and group_membership is not None
        return weighted_deepwalk.CrossWalkDeepWalk(
            num_nodes=num_nodes,
            group_membership=group_membership,
            weighted_adj=weighted_adj_path,
            embedding_dim=embedding_dim,
        )
    elif fairwalk:
        return weighted_deepwalk.FairWalkDeepWalk(
            num_nodes=num_nodes,
            group_membership=group_membership,
            embedding_dim=embedding_dim,
            weighted_adj=weighted_adj_path
        )
    return weighted_deepwalk.UnWeightedDeepWalk(
            num_nodes=num_nodes,
            embedding_dim=embedding_dim,
            weighted_adj=weighted_adj_path,
        )


def _get_node2vec_model(embedding_dim, num_nodes, weighted_adj_path=None, group_membership=None, crosswalk=True, fairwalk=False):
    assert not (crosswalk and fairwalk), "Both crosswalk and fairwalk cannot be true"
    if crosswalk:
        # assert weighted_adj_path is not None and group_membership is not None
        return weighted_node2vec.WeightedNode2Vec(
            num_nodes=num_nodes,
            group_membership=group_membership,
            weighted_adj=weighted_adj_path,
            embedding_dim=embedding_dim,
        )
    elif fairwalk:

        return weighted_node2vec.FairWalkNode2Vec(
            num_nodes=num_nodes,
            group_membership=group_membership,
            embedding_dim=embedding_dim,
            weighted_adj=weighted_adj_path
        )
    return weighted_node2vec.UnWeightedNode2Vec(
            num_nodes=num_nodes,
            embedding_dim=embedding_dim,
            weighted_adj=weighted_adj_path
        )


def get_feature_trained_get_embs(file_path):
    return torch.from_numpy(np.load(file_path).astype(np.float32))


def train_node2vec_get_embs(file_path, **kwargs):
    model = _get_node2vec_model(**kwargs)
    return model.train_and_get_embs(file_path)

def train_deepwalk_get_embs(file_path, **kwargs):
    model = _get_deepwalk_model(**kwargs)
    return model.train_and_get_embs(file_path)

def store_weighted_adj(file_path, edge_index, num_nodes, crosswalk, fairwalk, group_membership,):
    # make this edge index symmetric
    edge_index = torch.unique(torch.cat([edge_index, edge_index.flip(0)], dim=1), dim=1)
    row, col = edge_index
    from torch_sparse import SparseTensor
    adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    row, col, _ = adj.coo()
    ones = np.ones(row.shape[0], dtype=np.int32)
    A = sparse.csr_matrix((ones, (row.numpy(), col.numpy())), shape=(num_nodes, num_nodes))
    adj = get_reweighted_graph(A, crosswalk=crosswalk, fairwalk=fairwalk, group_membership=group_membership)
    sparse.save_npz(file_path, adj)

def get_num_nodes_from_adj(adj_path):
    return sparse.load_npz(adj_path).shape[0]

def get_adj_mat_from_path(adj_path):
    return sparse.load_npz(adj_path)


def get_edge_index_from_sparse_path(weighted_adj):
    adj = weighted_adj if sparse.issparse(weighted_adj) else sparse.load_npz(weighted_adj)
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

def get_gnn_model(model_name, num_features, emb_dim, dataset=None, num_layers=None, learn_outvec=True):
    if num_layers is None:
        assert dataset is not None, "If num_layers is not provided, dataset must be provided"
        num_layers = NUM_GNN_LAYERS[dataset]
    from utils.link_prediction import GATLinkPrediction, GCNLinkPrediction
    if model_name == 'gcn':
        model = GCNLinkPrediction(in_channels=num_features, embedding_size=emb_dim, hidden_channels=64, num_layers=num_layers, num_embeddings=num_features, learn_outvec=learn_outvec)
    elif model_name == 'gat':
        model = GATLinkPrediction(in_channels=num_features, embedding_size=emb_dim, hidden_channels=64, num_layers=num_layers, num_embeddings=num_features, learn_outvec=learn_outvec)
    else:
        raise NotImplementedError
    return model

def train_residual2vec(adj, noise_sampler, dim=128, batch_size=256, ):
    from node2vec import node2vecs
    from utils import graph_utils, config
    return graph_utils.generate_embedding_with_word2vec(A=adj, dim=dim, noise_sampler=noise_sampler, batch_size=batch_size, device=config.DEVICE)
    
def train_model_and_get_embs(adj, model_name, X, sampler, gnn_layers, epochs, learn_outvec, num_workers, batch_size, learning_rate, model_dim=128):
    if model_name == 'residual2vec':
        return train_residual2vec(adj=adj, noise_sampler=sampler, dim=model_dim, batch_size=batch_size)
    num_nodes = adj.shape[0]
    edge_index = get_edge_index_from_sparse_path(adj)
    print(edge_index.shape)
    from dataset.triplet_dataset import TripletGraphDataset, NeighborEdgeSampler
    dataset = TripletGraphDataset(
        X=X,
        edge_index=edge_index,
        sampler=sampler
    )
    dataloader = NeighborEdgeSampler(dataset, batch_size=256, shuffle=True, num_workers=num_workers, pin_memory=True)
    model = get_gnn_model(model_name=model_name, emb_dim=model_dim, num_layers=gnn_layers, num_features=X.shape[1], learn_outvec=learn_outvec)
    from residual2vec.residual2vec_sgd import residual2vec_sgd as rv
    frame = rv(noise_sampler=False, window_length=5, num_walks=10, walk_length=80, batch_size=batch_size).fit()
    frame.transform(model=model, dataloader=dataloader, epochs=epochs, learning_rate=learning_rate)
    embs = torch.zeros((num_nodes, model_dim))
    model.eval()
    dataloader = NeighborEdgeSampler(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, transforming=True)
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Generating node embeddings")):
            a, _, _ = batch
            a = model.forward_i(a).detach().cpu()
            nodes_remaining = num_nodes - (idx * batch_size)
            if nodes_remaining < batch_size:
                embs[idx * batch_size:, :] = a[-nodes_remaining:, :]
                break
            else:
                embs[idx * batch_size:(idx + 1) * batch_size, :] = a
    return embs

def get_reweighted_graph(adj, crosswalk, fairwalk, group_membership=None):
    from utils import graph
    from graph_embeddings import Fairwalk as fw
    assert adj.shape[0] == adj.shape[1] and not (crosswalk and fairwalk) and sparse.issparse(adj)
    # check symmetry
    assert (adj != adj.T).nnz == 0, "Adj matrix is not symmetric"
    A_ = adj.copy()
    if crosswalk:
        G = graph.from_numpy(adj, undirected=True)
        G.attr = group_membership
        n_groups = np.unique(group_membership).shape[0]
        graph.set_weights(adj.copy(), group_membership, G, exp_=2, p_bndry=.7, l=n_groups)
        A_ = graph.edge_weights_to_sparse(G, adj)
    if fairwalk:
        G = fw(group_membership=group_membership)
        A_ = G._get_adj_matrix(adj)
    return A_



def get_embs_from_dataset(dataset_name: str='generic', crosswalk: bool=False, r2v: bool=True, node2vec: bool=False, fairwalk: bool=False, model_name: str=None, learn_outvec:bool=False, model_dim=128, edges=None, group_membership=None, degree_agnostic=True):
    """
    returns embs given dataset name
    dataset: name of dataset
    crosswalk: either to use crosswalk or not
    r2v: either to use sbm sampling or not
    node2vec: method to use to generate node features True=node2vec False: deepwalk
    fairwalk: either to use fairwalk
    model_name: name of model to use, can be ['gcn', 'gat']
    """
    assert not (crosswalk and fairwalk)
    assert dataset_name in ['airport', 'polbook', 'polblog', 'facebook', 'twitch', 'generic']
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb: 512"
    os.environ['DISABLE_WANDB'] = "true"
    os.environ['DISABLE_TQDM'] = "true"
    dataset = get_dataset(dataset_name, edge_index=edges, group_membership=group_membership)
    group_membership = dataset.get_grouped_col()
    edge_index, num_nodes = dataset.edge_index, group_membership.shape[0]
    # probably we don't need this, but better to be safe
    edge_index = torch.unique(torch.cat([edge_index, edge_index.flip(0)], dim=1), dim=1)
    return_features = (crosswalk or fairwalk) or model_name in ['deepwalk', 'node2vec']
    num_features = model_dim
    adj = get_adj_from_edge_index(edge_index, num_nodes)
    # create data to train
    if node2vec:
        # using node2vec node features
        feature_model = _get_node2vec_model(embedding_dim=num_features, num_nodes=num_nodes, crosswalk=crosswalk, fairwalk=fairwalk, group_membership=group_membership, weighted_adj_path=adj)
    else:
        # use deepwalk node features
        feature_model = _get_deepwalk_model(embedding_dim=num_features, num_nodes=num_nodes, crosswalk=crosswalk, fairwalk=fairwalk, group_membership=group_membership, weighted_adj_path=adj)

    # weighted adj matrix
    adj = get_reweighted_graph(adj=adj, crosswalk=crosswalk, fairwalk=fairwalk, group_membership=group_membership)
    # train and get embs
    X = feature_model.train_and_get_embs(save=None).astype(np.float32)
    if return_features:
        return X
    assert not (crosswalk and fairwalk)
    assert model_name in ['gcn', 'gat', 'residual2vec']
    X = torch.from_numpy(X)
    
    # use weights in case of crosswalk or fairwalk
    use_weights = True if (crosswalk or fairwalk) else False
    # select sampler
    if model_name == 'residual2vec':
        from node2vec import node2vecs
        sampler = node2vecs.utils.node_sampler.SBMNodeSampler(group_membership=group_membership, window_length=1, dcsbm=not degree_agnostic)
    elif r2v:
        from dataset.triplet_dataset import SbmSamplerWrapper
        sbm_sampler_wrapper = SbmSamplerWrapper(adj_path=adj, group_membership=group_membership, window_length=1, padding_id=num_nodes, num_walks=100, num_edges=edge_index.shape[1], use_weights=use_weights, dcsbm=not degree_agnostic)
        sampler = sbm_sampler_wrapper.sample_neg_edges
    else:
        from torch_geometric.utils import negative_sampling
        sampler = negative_sampling
    
    
        
    return train_model_and_get_embs(adj=adj, model_name=model_name, X=X, sampler=sampler, gnn_layers=NUM_GNN_LAYERS[dataset_name], epochs=R2V_TRAINING_EPOCHS[dataset_name], learn_outvec=learn_outvec, num_workers=NUM_THREADS[dataset_name], batch_size=DATASET_BATCH_SIZE[dataset_name], learning_rate=DATASET_LEARNING_RATE[dataset_name], model_dim=model_dim)


def get_connected_components(dataset, get_labels=False):
    from scipy.sparse.csgraph import connected_components
    # get adj matrix from edge_index
    group_membership = dataset.get_grouped_col()
    adj = get_adj_from_edge_index(dataset.edge_index, group_membership.shape[0])
    components, labels = connected_components(adj, directed=False, return_labels=True)
    if get_labels:
        return components, labels
    return components

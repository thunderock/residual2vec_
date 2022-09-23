import os
from os.path import join as j
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
    def node2vec_weights(self):
        if self.crosswalk:
            return str(j(self.root, "{}_crosswalk_node2vec.h5".format(self.basename)))
        else:
            return str(j(self.root, "{}_node2vec.h5".format(self.basename)))

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
    # add other datasets here
    return dataset

def _get_node2vec_model(crosswalk, embedding_dim, num_nodes,walk_length, context_size,
                        edge_index, weighted_adj_path=None, group_membership=None):
    if crosswalk:
        # assert weighted_adj_path is not None and group_membership is not None
        return weighted_node2vec.WeightedNode2Vec(
            num_nodes=num_nodes,
            group_membership=group_membership,
            weighted_adj=weighted_adj_path,
            edge_index=edge_index,
            embedding_dim=embedding_dim,
            walk_length=walk_length,
            context_size=context_size,

        ).to(DEVICE)

    return weighted_node2vec.UnWeightedNode2Vec(
            num_nodes=num_nodes,
            edge_index=edge_index,
            embedding_dim=embedding_dim,
            walk_length=walk_length,
            context_size=context_size,
            weighted_adj=weighted_adj_path
        ).to(DEVICE)

def get_node2vec_trained_get_embs(file_path, **kwargs):
    model = _get_node2vec_model(**kwargs)
    model.load_state_dict(torch.load(file_path, map_location=DEVICE))
    return model.embedding.weight.detach().cpu()


def train_node2vec_get_embs(file_path, batch_size, num_workers, epochs, **kwargs):
    model = _get_node2vec_model(**kwargs)
    loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    return model.train_and_get_embs(loader, optimizer, epochs, file_path)

def store_crosswalk_weights(file_path, **kwargs):
    model = _get_node2vec_model(**kwargs)
    sparse.save_npz(file_path, model.weighted_adj)

def get_num_nodes_from_adj(adj_path):
    return sparse.load_npz(adj_path).shape[0]

def get_edge_index_from_sparse_path(weighted_adj):
    adj = sparse.load_npz(weighted_adj)
    row, col = adj.nonzero()
    return torch.cat((torch.from_numpy(row).unsqueeze(dim=0), torch.from_numpy(col).unsqueeze(dim=0))).long()

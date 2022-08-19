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


class TripletPokecDataset(Dataset):
    def __init__(self, root='/tmp/', ): # neighbors_sample_size=NUM_NEIGHBORS):
        self.root = root
        download_url("https://snap.stanford.edu/data/soc-pokec-profiles.txt.gz", self.root)
        download_url("https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz", self.root)
        node_fields = [
            "public",
            "completion_percentage",
            "gender",
            "region",
            "last_login",
            "registration",
            "AGE",
            "body",
            "I_am_working_in_field",
            "spoken_languages",
            "hobbies",
            "I_most_enjoy_good_food",
            "pets",
            "body_type",
            "my_eyesight",
            "eye_color",
            "hair_color",
            "hair_type",
            "completed_level_of_education",
            "favourite_color",
            "relation_to_smoking",
            "relation_to_alcohol",
            "sign_in_zodiac",
            "on_pokec_i_am_looking_for",
            "love_is_for_me",
            "relation_to_casual_sex",
            "my_partner_should_be",
            "marital_status",
            "children",
            "relation_to_children",
            "I_like_movies",
            "I_like_watching_movie",
            "I_like_music",
            "I_mostly_like_listening_to_music",
            "the_idea_of_good_evening",
            "I_like_specialties_from_kitchen",
            "fun",
            "I_am_going_to_concerts",
            "my_active_sports",
            "my_passive_sports",
            "profession",
            "I_like_books",
            "life_style",
            "music",
            "cars",
            "politics",
            "relationships",
            "art_culture",
            "hobbies_interests",
            "science_technologies",
            "computers_internet",
            "education",
            "sport",
            "movies",
            "travelling",
            "health",
            "companies_brands",
            "more",
            ""
        ]
        raw_paths = [self.root + "soc-pokec-profiles.txt.gz", self.root + "soc-pokec-relationships.txt.gz"]
        feature_cols = ["gender", "AGE", "public", "completion_percentage", "region"]
        dfn = pd.read_csv(raw_paths[0], sep="\t", names=node_fields, nrows=None)[feature_cols]
        dfe = pd.read_csv(raw_paths[1], sep="\t", names=["source", "target"], nrows=None)
        dfn["age"] = dfn["AGE"].copy()
        dfn = dfn.drop(columns=["AGE"])
        mx_freq_age = dfn["age"].value_counts().idxmax()
        mx_freq_gender = dfn.gender.value_counts().idxmax()
        dfn.age = dfn.age.fillna(mx_freq_age)
        dfn.gender = dfn.gender.fillna(mx_freq_gender)
        dfn.region = LabelEncoder().fit_transform(dfn.region)
        feature_cols[1] = "age"

        dfn = dfn.astype({'gender': np.float32, 'age': np.float32, 'public': np.float32, 'completion_percentage': np.float32, 'region': np.float32}).sort_index()
        # assert dfn.shape[0] == dfn.index.unique().shape[0]

        self.X = torch.cat([torch.from_numpy(dfn[col].values.reshape(-1, 1)) for col in feature_cols], dim=1, )
        dfe = dfe.astype({'source': 'int', 'target': 'int'})
        dfe = dfe.drop_duplicates()
        self.num_features = self.X.shape[1]

        # -1 because of we want to node ids to start from 0
        dfe = dfe[dfe.source != dfe.target] - 1
        self.edge_index = torch.cat([torch.from_numpy(dfe[col].values.reshape(-1, 1)) for col in ["source", "target"]], dim=1).T
        self.neg_edge_index = negative_sampling(edge_index=self.edge_index, num_nodes=self.X.shape[0], num_neg_samples=None, method='sparse', force_undirected=True)

    def __len__(self):
        # assumes that all the nodes ids are present starting from 0 to the max number of nodes
        return self.X.shape[0]

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
        if nodes.shape[0] == 0:
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
    def __init__(self, dataset, egde_sample_size=None, **kwargs):
        # investigate dual calling behaviour here, ideal case is calling this class with node_id range dataset
        # node_idx = torch.arange(self.adj_t.sparse_size(0))

        super().__init__(dataset=dataset, collate_fn=self.sample, **kwargs)
        self.features = self.dataset.num_features
        edge_index = dataset.edge_index
        num_nodes = len(dataset)
        self.features = dataset.num_features
        self.adj_t = self._get_adj_t(edge_index, num_nodes)
        self.neg_adj_t = self._get_adj_t(dataset.neg_edge_index, num_nodes)
        self.neg_adj_t.storage.rowptr()
        self.adj_t.storage.rowptr()
        if not egde_sample_size:
            egde_sample_size = num_nodes // 2
        self.edge_sample_size = torch.tensor(egde_sample_size)

    def _get_adj_t(self, edge_index, num_nodes):
        return SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.arange(edge_index.size(1)), sparse_sizes=(num_nodes, num_nodes)).t()

    def sample(self, batch):
        # make sure this is a list of tensors
        batch = torch.stack(batch)
        a, p, n = batch[:, 0], batch[:, 1], batch[:, 2]
        adjs, nids = [], []
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
        return (self.dataset.X[nids[0]], adjs[0]), \
               (self.dataset.X[nids[1]], adjs[1]), \
               (self.dataset.X[nids[2]], adjs[2])

    def __repr__(self) -> str:
        return '{}({}, batch_size={})'.format(self.__class__.__name__, len(self.dataset), self.batch_size)

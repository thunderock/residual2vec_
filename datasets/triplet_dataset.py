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

class TripletPokecDataset(Dataset):
    def __init__(self, root='/tmp/',):
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

        dfn = dfn.astype({'gender': 'float', 'age': 'float', 'public': 'float', 'completion_percentage': 'float', 'region': 'float'}).sort_index()
        assert dfn.shape[0] == dfn.index.unique().shape[0]
        # -1 because of we want to node ids to start from 0 because negative sampling generates node ids starting from 0
        # max_node_id = (dfn.index.values - 1).max()
        # self.idx_mapping = torch.full((max_node_id + 1,), -1, dtype=torch.long)
        # idx_mapping = {idx: i for i, idx in enumerate(dfn.index.values - 1)}
        # for k, v in idx_mapping.items():
        #     self.idx_mapping[k] = v
        # assert (self.idx_mapping == -1).sum() == 0
        self.X = torch.cat([torch.from_numpy(dfn[col].values.reshape(-1, 1)) for col in feature_cols], dim=1)
        dfe = dfe.astype({'source': 'int', 'target': 'int'})
        dfe = dfe.drop_duplicates()

        # -1 because of we want to node ids to start from 0
        dfe = dfe[dfe.source != dfe.target] - 1
        # dfe = dfe[(dfe.source.isin(idx_mapping)) & (dfe.target.isin(idx_mapping))] - 1
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

    def _get_features_for_node(self, idx):
        # index = self.idx_mapping[idx]
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
        # print(type(idx))
        # select a node with positive edge
        p_node = self._select_random_neighbor(idx)
        n_node = self._select_random_neighbor(idx, neg=True)
        if p_node and n_node:
            return self._get_features_for_node(idx), self._get_features_for_node(p_node), self._get_features_for_node(n_node), torch.tensor([1])

        idx = torch.randint(self.edge_index.shape[0], (1,))
        # print("randomly selected a_idx", idx)
        # idx = 102
        # select nodes randomly here
        a = self._get_features_for_node(self.edge_index[0, idx].item())
        # p cannot be none now
        p = self._get_features_for_node(self._select_random_neighbor(idx))
        # this can still result in failure but haven't seen it yet, this means that negative sampling couldn't generate a negative node for this source node
        n = self._get_features_for_node(self._select_random_neighbor(idx, neg=True))
        return a, p, n, torch.tensor([0])




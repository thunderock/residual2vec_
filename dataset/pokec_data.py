# @Filename:    pokec_data.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        9/1/22 5:45 PM

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch_geometric.data import download_url

class PokecDataFrame(object):

    def __init__(self, group_col: str = 'gender', root: str = '/tmp/'):
        download_url("https://snap.stanford.edu/data/soc-pokec-profiles.txt.gz", root)
        download_url("https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz", root)
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
        raw_paths = [root + "soc-pokec-profiles.txt.gz", root + "soc-pokec-relationships.txt.gz"]
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
        self.X = torch.cat([torch.from_numpy(dfn[col].values.reshape(-1, 1)) for col in feature_cols], dim=1, )
        dfe = dfe.astype({'source': 'int', 'target': 'int'})
        dfe = dfe.drop_duplicates()
        dfe = dfe[dfe.source != dfe.target] - 1
        self.edge_index = torch.cat([torch.from_numpy(dfe[col].values.reshape(-1, 1)) for col in ["source", "target"]],
                                    dim=1).T
        self.group_col = feature_cols.index(group_col)

    def get_grouped_col(self):
        assert self.group_col is not None, "Group column not specified"
        return self.X[:, self.group_col]
# @Filename:    polblog_data.py
# @Author:      Sadamori kojaku
# @Email:       skojaku@gmail.com
# @Time:        9/14/22

import numpy as np
import pandas as pd
import torch

class PolBlogDataFrame(object):

    def __init__(self, group_col: str = 'political_leaning', root: str = '/tmp/'):

        # Setting
        node_feature_url = "https://raw.githubusercontent.com/skojaku/core-periphery-detection/master/data/ent.moreno_blogs_blogs.blog.orientation"
        edge_list_url = "https://raw.githubusercontent.com/skojaku/core-periphery-detection/master/data/out.moreno_blogs_blogs"
        feature_cols = ["political_leaning"]

        # Download
        dfn = pd.read_csv(node_feature_url, header=None, names = ["political_leaning"])
        dfe = pd.read_csv(edge_list_url)
        # Clearn up and uniqify the categorical values
        dfn["political_leaning"] = np.unique(dfn["political_leaning"].values, return_inverse=True)[1]
        dfe = dfe.astype({'source': 'int', 'target': 'int'})
        dfe = dfe.drop_duplicates()
        dfe = dfe[dfe.source != dfe.target] - 1

        # Create the membership variables
        self.X = torch.cat([torch.from_numpy(dfn[col].values.reshape(-1, 1)) for col in feature_cols], dim=1, )
        self.edge_index = torch.cat([torch.from_numpy(dfe[col].values.reshape(-1, 1).astype(np.int32)) for col in ["source", "target"]],dim=1).T
        self.group_col = feature_cols.index(group_col)

    def get_grouped_col(self):
        assert self.group_col is not None, "Group column not specified"
        return self.X[:, self.group_col]
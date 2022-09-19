# @Filename:    airportnet_data.py
# @Author:      Sadamori kojaku
# @Email:       skojaku@gmail.com
# @Time:        9/14/22
from scipy import sparse
import numpy as np
import pandas as pd
import torch
import networkx as nx
import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import networkx as nx


class AirportNetDataFrame(object):

    def __init__(self, group_col: str = 'region', root: str = '/tmp/'):

        # Setting
        A, labels, node_table = load_airport_net()
        group_ids = np.unique(labels, return_inverse=True)[1]

        feature_cols = ["region"]

        # Download
        # Clearn up and uniqify the categorical values
        dfn = pd.DataFrame({"region":group_ids})
        r, c, _ = sparse.find(A)
        dfe = pd.DataFrame({"source":r, "target":c})
        dfn["region"] = np.unique(dfn["region"].values, return_inverse=True)[1]
        dfe = dfe.astype({'source': 'int', 'target': 'int'})
        dfe = dfe.drop_duplicates()
        dfe = dfe[dfe.source != dfe.target]

        # Create the membership variables
        self.X = torch.cat([torch.from_numpy(dfn[col].values.reshape(-1, 1)) for col in feature_cols], dim=1, )
        self.edge_index = torch.cat([torch.from_numpy(dfe[col].values.reshape(-1, 1).astype(np.int32)) for col in ["source", "target"]],dim=1).T
        self.group_col = feature_cols.index(group_col)

    def get_grouped_col(self):
        assert self.group_col is not None, "Group column not specified"
        return self.X[:, self.group_col]

def load_network(func):
    def wrapper(binarize=True, symmetrize=False, *args, **kwargs):
        net, labels, node_table = func(*args, **kwargs)
        if symmetrize:
            net = net + net.T
            net.sort_indices()

        _, comps = connected_components(csgraph=net, directed=False, return_labels=True)
        ucomps, freq = np.unique(comps, return_counts=True)
        s = comps == ucomps[np.argmax(freq)]
        labels = labels[s]
        net = net[s, :][:, s]
        if binarize:
            net = net + net.T
            net.data = net.data * 0 + 1
        node_table = node_table[s]
        return net, labels, node_table

    return wrapper


@load_network
def load_airport_net():
    # Node attributes
    node_table = pd.read_csv(
        "https://raw.githubusercontent.com/skojaku/core-periphery-detection/master/data/node-table-airport.csv"
    )

    # Edge table
    edge_table = pd.read_csv(
        "https://raw.githubusercontent.com/skojaku/core-periphery-detection/master/data/edge-table-airport.csv"
    )
    # net = nx.adjacency_matrix(nx.from_pandas_edgelist(edge_table))

    net = sparse.csr_matrix(
        (
            edge_table["weight"].values,
            (edge_table["source"].values, edge_table["target"].values),
        ),
        shape=(node_table.shape[0], node_table.shape[0]),
    )

    s = ~pd.isna(node_table["region"])
    node_table = node_table[s]
    labels = node_table["region"].values
    net = net[s, :][:, s]
    return net, labels, node_table

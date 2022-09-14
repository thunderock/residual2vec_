# @Filename:    polblog_data.py
# @Author:      Sadamori kojaku
# @Email:       skojaku@gmail.com
# @Time:        9/14/22
from scipy import sparse
import numpy as np
import pandas as pd
import torch
import networkx as nx
import os.path as osp
import ssl
import sys
import urllib
from typing import Optional

class PolBookDataFrame(object):

    def __init__(self, group_col: str = 'political_leaning', root: str = '/tmp/'):

        # Setting
        net_file = "https://raw.githubusercontent.com/skojaku/residual2vec/main/notebooks/2022_07_17_graph_reconstruction/data/polbooks.gml"

        download_url(net_file, root)

        G = nx.read_gml(root + "polbooks.gml")
        G = nx.relabel.convert_node_labels_to_integers(
            G, first_label=0, ordering="default"
        )  # first_label is the starting integer label, in this case zero
        nodes = G.nodes(data=True)
        labels, group_ids = np.unique([nd[1]["value"] for nd in nodes], return_inverse=True)
        A = nx.adjacency_matrix(G).asfptype()

        feature_cols = ["political_leaning"]

        # Download
        # Clearn up and uniqify the categorical values
        dfn = pd.DataFrame({"political_leaning":group_ids})
        r, c, _ = sparse.find(A)
        dfe = pd.DataFrame({"source":r, "target":c})
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

def download_url(url: str, folder: str, log: bool = True,
                 filename: Optional[str] = None):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    if filename is None:
        filename = url.rpartition('/')[2]
        filename = filename if filename[0] == '?' else filename.split('?')[0]

    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print(f'Using existing file {filename}', file=sys.stderr)
        return path

    if log:
        print(f'Downloading {url}', file=sys.stderr)

    #makedirs(folder)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as f:
        # workaround for https://bugs.python.org/issue42853
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path
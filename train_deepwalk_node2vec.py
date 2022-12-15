# @Filename:    train_deepwalk.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        12/15/22 5:32 PM

import numpy as np
from utils import snakemake_utils
import warnings
warnings.filterwarnings("ignore")


numbers = ['one', 'two', 'three', 'four', 'five']
dim = 128




def get_embs(adj_file_paths, dataset, method):
    for number, file_path in zip(numbers, adj_file_paths):
        adj_mat = snakemake_utils.get_adj_mat_from_path(file_path)
        print(adj_mat.shape, file_path, method)
        if method == 'deepwalk':
            feature_model = snakemake_utils._get_deepwalk_model(embedding_dim=dim, num_nodes=adj_mat.shape[0], edge_index=None, crosswalk=False, fairwalk=False, group_membership=None, weighted_adj_path=adj_mat)
        else:
            feature_model = snakemake_utils._get_node2vec_model(embedding_dim=dim, num_nodes=adj_mat.shape[0], edge_index=None, crosswalk=False, fairwalk=False, group_membership=None, weighted_adj_path=adj_mat)
        embs = feature_model.train_and_get_embs(save=None)
        np.save('../final_crosswalk/{}/{}_{}/{}_{}_{}_embs.npy'.format(dataset, dataset, number, dataset, method, dim), embs)

for method in ['deepwalk', 'node2vec']:
    assert method in ['deepwalk', 'node2vec']
    datasets = ['airport', 'polbook', 'polblog', 'pokec']
    for dataset in datasets:
        adj_file_paths = ['../final_crosswalk/{}/{}_{}/{}_adj.npz'.format(dataset, dataset, number, dataset) for number in numbers]
        get_embs(adj_file_paths, dataset, method)

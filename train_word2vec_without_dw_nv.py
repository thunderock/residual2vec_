import numpy as np
from utils import snakemake_utils, graph_utils
from node2vec import node2vecs


device = "cuda:0"
numbers = ['one']# , 'two', 'three', 'four', 'five']
dim = 128
datasets = ['airport', 'polbook', 'polblog', 'pokec']
datasets = ['airport']
method = "word2vec"

for dataset in datasets:
    adj_file_paths = ['../final_crosswalk/{}/{}_{}/{}_adj.npz'.format(dataset, dataset, number, dataset) for number in numbers]

    for idx, adj_file_path in enumerate(adj_file_paths):
        number = numbers[idx]
        adj_mat = snakemake_utils.get_adj_mat_from_path(adj_file_path)
        # convert this to csr format
        adj_mat = adj_mat.tocsr()
        
        noise_sampler = node2vecs.utils.node_sampler.ConfigModelNodeSampler()
        embs = graph_utils.generate_embedding_with_word2vec(adj_mat, dim, noise_sampler, device)
        
        np.save('../final_crosswalk/{}/{}_{}/{}_{}_{}_embs.npy'.format(dataset, dataset, number, dataset, method, dim), embs)

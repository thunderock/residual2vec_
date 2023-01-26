import numpy as np
from utils import snakemake_utils, graph_utils
from node2vec import node2vecs
import torch.multiprocessing as mp


device = "cuda:1"
numbers = ['one' , 'two', 'three', 'four', 'five']
dim = 128
datasets = ['airport' , 'polblog', 'polbook', 'pokec']
method = "residual2vec"

def task(dataset, number):
    adj_file_path = '../final_crosswalk_temp/{}/{}_{}/{}_adj.npz'.format(dataset, dataset, number, dataset)
    adj_mat = snakemake_utils.get_adj_mat_from_path(adj_file_path)
    adj_mat = adj_mat.tocsr()
    print(dataset, number, adj_mat.shape)

    group_ids = snakemake_utils.get_dataset(dataset).get_grouped_col().numpy()
    noise_sampler = node2vecs.utils.node_sampler.SBMNodeSampler(group_membership=group_ids, window_length=1)
    embs = graph_utils.generate_embedding_with_word2vec(adj_mat, dim, noise_sampler, device)

    np.save('../final_crosswalk_temp/{}/{}_{}/{}_{}_{}_embs.npy'.format(dataset, dataset, number, dataset, method, dim), embs)
    return True

def divide_chunks(l, n):
     
    for i in range(0, len(l), n):
        yield l[i:i + n]
if __name__ == '__main__':
    params = [(dataset, number) for number in numbers for dataset in datasets]
    
    params = list(divide_chunks(params, 3))
    num_processes = 3
    processes = []
    mp.set_start_method('spawn')
    for param in params:
        for rank in range(num_processes):
            p = mp.Process(target=task, args=param[rank])
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

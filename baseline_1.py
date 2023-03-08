# -*- coding: utf-8 -*-
# @Author: Ashutosh Tiwari
# @Email: checkashu@gmail.com
# @Date:   2023-02-13 11:09:50


import sys
from utils import graph_utils, snakemake_utils
from models import fast_knn_cpu
import numpy as np
from sklearn.decomposition import PCA
from baseline.debias_graph import debias_wrapper
from baseline.we import doPCA
from baseline.we_utils import get_direction
from os.path import join as j


if "snakemake" in sys.modules:
    BASE = snakemake.params["DATA_ROOT"]
    NODE2VEC = snakemake_utils.get_string_boolean(snakemake.params["NODE2VEC"])
    DATASET = snakemake.params["DATASET"]
else:
    BASE = "../final_/polbook/polbook_one/"
    NODE2VEC = False
    DATASET = "polbook"

METHOD_NAME = 'node2vec' if NODE2VEC else 'deepwalk'
print("BASE: ", BASE, "NODE2VEC: ", NODE2VEC, "DATASET: ", DATASET)

def get_embs(dataset, node2vec=NODE2VEC):
    y = snakemake_utils.get_dataset(dataset).get_grouped_col().numpy()
    deepwalk = np.load(j(BASE, "{}_{}.npy".format(dataset, METHOD_NAME)))
    
    
    centroids = graph_utils.get_centroid_per_group(deepwalk, y)
    # definitional words, these are supposed to be represent the group,
    # in this case lets take these to be the nodes closest to centroid of group
    # in this case are the centroids of the groups
    definitional = graph_utils.get_n_nearest_neighbors_for_nodes(
        nodes=centroids, 
        embs=deepwalk,
        k=1,
        metric='cosine'
    )
    
    N, dim = deepwalk.shape
    K = np.unique(y).shape[0]
    
    gender_specific_nodes = graph_utils.get_n_nearest_neighbors_for_nodes(
        nodes=centroids, 
        embs=deepwalk,
        k=int (.2 * N) // K,
        metric='cosine'
    )
    equalize = graph_utils.get_farthest_pairs(deepwalk, y, same_class=False, 
                                              per_class_count=int((.2 * N) / K))
    
    print("number of equalize pairs: ", equalize.shape)
    direction = get_direction(deepwalk, y, "PCA")
    
    return debias_wrapper(emb=deepwalk,gender_specific_words=gender_specific_nodes, 
               definitional=None, equalize=equalize, y=y, direction=direction,
               drop_gender_specific_words=True)
            

embs = get_embs(dataset=DATASET, node2vec=NODE2VEC)
np.save(j(BASE, "{}_baseline_man_woman+{}_embs.npy".format(DATASET, METHOD_NAME)), embs)


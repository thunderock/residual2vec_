# -*- coding: utf-8 -*-
# @Author: Ashutosh Tiwari
# @Email: checkashu@gmail.com
# @Date:   2023-02-24 14:40:10
# @Filepath: workflow/plot_fairness_per_node.py
import sys, os


DATASETS = ["polbook", "polblog", "airport", "twitch", "facebook"]
# print present working directory
BASE_DIR = "../../final_"
OUTPUT_FILE = "../figs/deepwalk_disparity_per_node.png"
EMBS_MAPPING = {
    "fairwalk+deepwalk": "_fairwalk_deepwalk.npy",
    "fairwalk+node2vec": "_fairwalk_node2vec.npy",
    "crosswalk+deepwalk": "_crosswalk_deepwalk.npy",
    "crosswalk+node2vec": "_crosswalk_node2vec.npy",
    "GCN+deepwalk+random": "_gcn_deepwalk_deepwalk_embs.npy",
    "GCN+deepwalk+r2v": "_gcn_deepwalk_r2v_embs.npy",
    "GCN+node2vec+random": "_gcn_node2vec_deepwalk_embs.npy",
    "GCN+node2vec+r2v": "_gcn_node2vec_r2v_embs.npy",
    "GAT+deepwalk+random": "_gat_deepwalk_deepwalk_embs.npy",
    "GAT+deepwalk+r2v": "_gat_deepwalk_r2v_embs.npy",
    "GAT+node2vec+random": "_gat_node2vec_deepwalk_embs.npy",
    "GAT+node2vec+r2v": "_gat_node2vec_r2v_embs.npy",
    "deepwalk": "_deepwalk.npy",
    "node2vec": "_node2vec.npy",
    "residual2vec": "_residual2vec_embs.npy",
    "baseline+deepwalk": "_baseline_man_woman+deepwalk_embs.npy",
    "baseline+node2vec": "_baseline_man_woman+node2vec_embs.npy",}
SAMPLE_ID = "one"

print(os.getcwd())
print(os.listdir())
if "snakemake" in sys.modules:
    DATASETS = snakemake.params["datasets"]
    BASE_DIR = snakemake.params["base_dir"]
    OUTPUT_FILE = str(snakemake.output)
    EMBS_MAPPING = snakemake.params["embs_mapping"]
    SAMPLE_ID = snakemake.params["sample_id"]
    print(DATASETS)
    print(BASE_DIR)
    print(OUTPUT_FILE)
    print(EMBS_MAPPING)
    print(SAMPLE_ID)
    sys.path.insert(0, '../residual2vec_')    

from os.path import join as j
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from utils import snakemake_utils, score

def get_embs(dataset):

    folder = j(BASE_DIR, dataset, dataset + '_' + SAMPLE_ID)
    ret = {}
    for i in EMBS_MAPPING.keys():
        ret[i] = np.load(j(folder, dataset + "_" + EMBS_MAPPING[i]))
    return ret

ARCHS = ["GCN", "GAT", "word2vec"]

arch_mapping = {
    "GCN": {
        "baseline": "GCN+deepwalk+random",
        "proposed": "GCN+deepwalk+r2v",
        },
    "GAT": {
        "baseline": "GAT+deepwalk+random",
        "proposed": "GAT+deepwalk+r2v",
        },
    "word2vec": {
        "baseline": "deepwalk",
        "proposed": "residual2vec",
        },
}

embs = []
N = []
y = []
for dataset in tqdm(DATASETS,desc="loading embs"):
    embs.append(get_embs(dataset))
    d = snakemake_utils.get_dataset(dataset)
    y.append(d.get_grouped_col().numpy())
    N.append(y[-1].shape[0])

df = []

for arch in ARCHS:
    

    


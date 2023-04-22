# -*- coding: utf-8 -*-
# @Author: Ashutosh Tiwari
# @Email: checkashu@gmail.com
# @Date:   2023-02-24 14:40:10
# @Filepath: workflow/plot_fairness_per_node.py
import sys, os


DATASETS = ["polbook", "polblog", "airport", 'twitch', 'facebook']
# print present working directory
BASE_DIR = "../final_"
OUTPUT_FILE = "figs/deepwalk_disparity_per_node.png"
CW_OUTPUT_FILE = "figs/crosswalk_disparity_per_node.png"
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
    "groupbiased+residual2vec": "_residual2vec_groupbiased_embs.npy",
    "baseline+deepwalk": "_baseline_man_woman+deepwalk_embs.npy",
    "baseline+node2vec": "_baseline_man_woman+node2vec_embs.npy",
    "groupbiased+gat+deepwalk": "_gat_deepwalk_r2v_groupbiased_embs.npy",
    "groupbiased+gat+node2vec": "_gat_node2vec_r2v_groupbiased_embs.npy",
    "groupbiased+gcn+deepwalk": "_gcn_deepwalk_r2v_groupbiased_embs.npy",
    "groupbiased+gcn+node2vec": "_gcn_node2vec_r2v_groupbiased_embs.npy",}
SAMPLE_IDS = ["one", "two", "three", "four", "five"]

print(os.getcwd())
print(os.listdir())
if "snakemake" in sys.modules:
    DATASETS = snakemake.params["datasets"]
    BASE_DIR = snakemake.params["base_dir"]
    OUTPUT_FILE = str(snakemake.output[0])
    CW_OUTPUT_FILE = str(snakemake.output[1])
    EMBS_MAPPING = snakemake.params["embs_mapping"]
    SAMPLE_IDS = snakemake.params["sample_ids"]
print(DATASETS)
print(BASE_DIR)
print(OUTPUT_FILE)
print(EMBS_MAPPING)
print(SAMPLE_IDS)
 
sys.path.insert(0, '../residual2vec_')    

from os.path import join as j
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from utils import snakemake_utils, score

def get_embs(dataset,sample_id):

    folder = j(BASE_DIR, dataset, dataset + '_' + sample_id)
    ret = {}
    for i in EMBS_MAPPING.keys():
        ret[i] = np.load(j(folder, dataset + EMBS_MAPPING[i]))
    return ret

ARCHS = ["GCN", "GAT", "word2vec", "crosswalk", "fairwalk"]

ARCH_MAPPING = {
    "GCN": {
        "baseline": "GCN+deepwalk+random",
        "proposed": "groupbiased+gcn+deepwalk",
        },
    "GAT": {
        "baseline": "GAT+deepwalk+random",
        "proposed": "groupbiased+gat+deepwalk",
        },
    "word2vec": {
        "baseline": "deepwalk",
        "proposed": "groupbiased+residual2vec",
        },
    "crosswalk": {
        "baseline": "deepwalk",
        "proposed": "crosswalk+deepwalk",
    },
    "fairwalk": {
        "baseline": "deepwalk",
        "proposed": "fairwalk+deepwalk",
    }
}

mp = {
    'Dataset': [],
    'architecture': [],
    'score': []
}

for dataset in tqdm(DATASETS,desc="loading embs"):
    y = snakemake_utils.get_dataset(dataset).get_grouped_col().numpy()
    df = []        
    for sample_id in SAMPLE_IDS:
        emb = get_embs(dataset, sample_id)
        for arch in ARCHS:
            for model in ["baseline", "proposed"]:
                s = score.get_node_parity(emb[ARCH_MAPPING[arch][model]], y, 'std')
                df.append(pd.DataFrame({
                    'dataset': dataset,
                    'disparity per node': s,
                    'model': model,
                    'architecture': arch,
                    'sample_id': sample_id,
                    "node_id": np.arange(len(y))
                }))              
    df = pd.concat(df, axis=0, ignore_index=True)
    for arch in ARCHS:
        baseline_scores = np.array(df[(df.architecture == arch) & (df.model == 'baseline') & (df.dataset == dataset)][['disparity per node', 'node_id']].groupby('node_id', sort=True)['disparity per node'].apply(list).tolist())
        proposed_scores = np.array(df[(df.architecture == arch) & (df.model == 'proposed') & (df.dataset == dataset)][['disparity per node', 'node_id']].groupby('node_id', sort=True)['disparity per node'].apply(list).tolist())
        
        ratios = ((baseline_scores - proposed_scores) > 0).sum(axis=0) / baseline_scores.shape[0]
        for idx, ratio in enumerate(ratios):
            
            mp['Dataset'].append(dataset.capitalize())
            mp['architecture'].append(arch)
            mp['score'].append(ratio)


fdf = pd.DataFrame(mp)

def plot_local_fairness(dframe, file_name):
    ax=sns.pointplot(data=dframe, x='Dataset', y='score', hue='architecture', palette='Set2', dodge=True, join=False, capsize=.15)
    plt.ylabel('Fraction of nodes debiased by \n proposed method', fontsize=15)
    plt.xlabel('Datasets', fontsize=20)
    ax.legend(loc="upper right", prop = { "size": 8 }, frameon=False)
    plt.axhline(y=.5, linestyle='--', c='#4D4D4D', )
    plt.xticks(list(plt.xticks()[0]) + [.5], fontsize=14)
    plt.yticks(fontsize=14)
    sns.despine()
    #save figure
    plt.savefig(file_name, dpi='figure', bbox_inches='tight')
    plt.close()

plot_local_fairness(fdf[~fdf.architecture.isin(['crosswalk', 'fairwalk'])], OUTPUT_FILE)


plot_local_fairness(fdf[fdf.architecture.isin(['crosswalk', 'fairwalk'])], CW_OUTPUT_FILE)

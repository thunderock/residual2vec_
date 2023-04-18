# -*- coding: utf-8 -*-
# @Author: Ashutosh Tiwari
# @Email: checkashu@gmail.com
# @Date:   2023-04-18 18:13:09
# @Filepath: workflow/deepwalk_global_fairness.py
import sys, os


DATASETS = ["polbook", "polblog", "airport", 'twitch', 'facebook']
# print present working directory
BASE_DIR = "../final_"
OUTPUT_FILE = "figs/deepwalk_global_fairness.png"
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
    OUTPUT_FILE = str(snakemake.output)
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def get_embs(dataset, sample_id):

    folder = j(BASE_DIR, dataset, dataset + '_' + sample_id)
    ret = {}
    for i in EMBS_MAPPING.keys():
        ret[i] = np.load(j(folder, dataset + EMBS_MAPPING[i]))
    return ret

def get_labels(dataset):
    return snakemake_utils.get_dataset(dataset).get_grouped_col().numpy()


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
}

ARCHS = ["GCN", "GAT", "word2vec"]



# arch, ds = "GCN", "polbook"

# embs = get_embs(ds, "one")
# proposed = embs[ARCH_MAPPING[arch]["proposed"]]
# baseline = embs[ARCH_MAPPING[arch]["baseline"]]
# y = get_labels(ds)


mp = {
    'Dataset': [],
    'architecture': [],
    'score': []
}



def get_fairness_score(emb, y_):
    k = np.unique(y_).shape[0]
    v = LinearDiscriminantAnalysis(n_components=k - 1).fit(emb, y_)
    v = v.scalings_
    v = np.einsum("ij,j->ij", v, 1 / np.linalg.norm(v, axis=0))
    num = np.var(emb @ v, axis=0)
    den = np.var(emb, axis=0)
    return np.sum(num) / np.sum(den)
    
# proposed = get_fairness_score(proposed, y)
# baseline = get_fairness_score(baseline, y)
# 
# print(proposed, baseline, proposed / baseline)

for dataset in tqdm(DATASETS,desc="loading embs"):
    y = snakemake_utils.get_dataset(dataset).get_grouped_col().numpy()
    df = []        
    for sample_id in SAMPLE_IDS:
        emb = get_embs(dataset, sample_id)
        for arch in ARCHS:
            for model in ["baseline", "proposed"]:
                s = get_fairness_score(emb[ARCH_MAPPING[arch][model]], y)
                df.append((
                    dataset,
                    s,
                    model,
                    arch,
                    sample_id
                ))              
    df = pd.DataFrame(df, columns=['dataset', 'disparity per node', 'model', 'architecture', 'sample_id'])
    df.to_csv(f"/tmp/df_{dataset}.csv", index=False)
    for arch in ARCHS:
        baseline_scores = np.mean(df[(df.architecture == arch) & (df.model == 'baseline') & (df.dataset == dataset)]['disparity per node'].values)
        proposed_scores = np.mean(df[(df.architecture == arch) & (df.model == 'proposed') & (df.dataset == dataset)]['disparity per node'].values)
        ratio = proposed_scores / baseline_scores
        mp['Dataset'].append(dataset.capitalize())
        mp['architecture'].append(arch)
        mp['score'].append(ratio)

fdf = pd.DataFrame(mp)

ax=sns.barplot(data=fdf, x='Dataset', y='score', hue='architecture', palette='Set2')

plt.xlabel('Datasets', fontsize=20)
ax.legend(loc="upper right", prop = { "size": 8 }, frameon=False)
plt.axhline(y=1., linestyle='--', c='red', )
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
sns.despine()
#save figure
plt.savefig(OUTPUT_FILE, dpi='figure', bbox_inches='tight')


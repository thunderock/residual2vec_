# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-01-18 09:05:36
# @Last Modified by:   Ashutosh Tiwari
# @Last Modified time: 2023-04-09 16:49:40
import seaborn as sns

model_names = {
    "fairwalk+deepwalk": "Fairwalk",
    "fairwalk+node2vec": "Fairwalk (node2vec)",
    "crosswalk+deepwalk": "CrossWalk",
    "crosswalk+node2vec": "CrossWalk (node2vec)",
    "GCN+deepwalk+random": "GCN",
    "GCN+deepwalk+r2v": "GCN (debiased)",
    "GCN+node2vec+random": "GCN-node2vec",
    "GCN+node2vec+r2v": "GCN-node2vec (debiased)",
    "GAT+deepwalk+random": "GAT",
    "GAT+deepwalk+r2v": "GAT (debiased)",
    "GAT+node2vec+random": "GAT-node2vec",
    "GAT+node2vec+r2v": "GAT-node2vec (debiased)",
    "deepwalk": "DeepWalk",
    "node2vec": "node2vec",
    "residual2vec": "DeepWalk (debiased)",
    "baseline+deepwalk": "Baseline (debiased)",
    "baseline+node2vec": "Baseline-node2vec (debiased)",
    "er_sampler+gat+node2vec": "GAT-node2vec",
    "er_sampler+gat+deepwalk": "GAT",
    "er_sampler+gcn+node2vec": "GCN-node2vec",
    "er_sampler+gcn+deepwalk": "GCN",
}

model2group = {
    "fairwalk+deepwalk": "Fairwalk",
    "fairwalk+node2vec": "Fairwalk",
    "crosswalk+deepwalk": "CrossWalk",
    "crosswalk+node2vec": "CrossWalk",
    "GCN+deepwalk+random": "GCN",
    "GCN+deepwalk+r2v": "GCN",
    "GCN+node2vec+random": "GCN",
    "GCN+node2vec+r2v": "GCN",
    "GAT+deepwalk+random": "GAT",
    "GAT+deepwalk+r2v": "GAT",
    "GAT+node2vec+random": "GAT",
    "GAT+node2vec+r2v": "GAT",
    "deepwalk": "DeepWalk",
    "node2vec": "DeepWalk",
    "residual2vec": "DeepWalk",
    "baseline+deepwalk": "Manipulation of embedding",
    "baseline+node2vec": "Manipulation of embedding",
    "er_sampler+gat+node2vec": "GAT",
    "er_sampler+gat+deepwalk": "GAT",
    "er_sampler+gcn+node2vec": "GCN",
    "er_sampler+gcn+deepwalk": "GCN"
}
model2type = {
    "fairwalk+deepwalk": "Debiased",
    "fairwalk+node2vec": "Debiased",
    "crosswalk+deepwalk": "Debiased",
    "crosswalk+node2vec": "Debiased",
    "GCN+deepwalk+random": "Vanilla",
    "GCN+deepwalk+r2v": "Debiased",
    "GCN+node2vec+random": "Vanilla",
    "GCN+node2vec+r2v": "Debiased",
    "GAT+deepwalk+random": "Vanilla",
    "GAT+deepwalk+r2v": "Debiased",
    "GAT+node2vec+random": "Vanilla",
    "GAT+node2vec+r2v": "Debiased",
    "deepwalk": "Vanilla",
    "node2vec": "Vanilla",
    "residual2vec": "Debiased",
    "baseline+deepwalk": "Debiased",
    "baseline+node2vec": "Debiased",
    "er_sampler+gat+node2vec": "Vanilla",
    "er_sampler+gat+deepwalk": "Vanilla",
}

model2markers = {
    "DeepWalk":"s",
    "GCN": "o",
    "GAT": "d",
    "Fairwalk": "v",
    "CrossWalk": "^",
    "Manipulation of embedding": "p",
}

model2linestyle = {
    "DeepWalk": "-",
    "GCN": "--",
    "GAT": ":",
    "Fairwalk": "-.",
    "CrossWalk": "-",
    "Manipulation of embedding": "-",
}


model_order = ["DeepWalk", "GCN", "GAT", "Fairwalk", "CrossWalk", "Manipulation of embedding"]


# ========================
# Color and Style
# ========================

mcmap = sns.color_palette().as_hex()
bcmap = sns.color_palette("bright").as_hex()
cmap = sns.color_palette("colorblind").as_hex()

model2colors = {
    "GCN": cmap[0],
    "GAT": cmap[1],
    "DeepWalk": cmap[3],
    "Fairwalk": "#2d2d2d",
    "CrossWalk": "#8d8d8d",
    "Manipulation of embedding": "red",
    
}

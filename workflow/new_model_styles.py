# -*- coding: utf-8 -*-
# @Author: Ashutosh Tiwari
# @Email: checkashu@gmail.com
# @Date:   2023-04-22 20:43:05
# @Filepath: workflow/new_model_styles.py


import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum


class Architecture(Enum):
    Word2Vec = 1
    GAT = 2
    GCN = 3


MODEL_TO_IS_ARCHITECTURE = {
    "fairwalk+deepwalk": False,
    "fairwalk+node2vec": False,
    "crosswalk+deepwalk": False,
    "crosswalk+node2vec": False,
    "GCN+deepwalk+random": Architecture.GCN,
    "GCN+deepwalk+r2v": Architecture.GCN,
    "GCN+node2vec+random": Architecture.GCN,
    "GCN+node2vec+r2v": Architecture.GCN,
    "GAT+deepwalk+random": Architecture.GAT,
    "GAT+deepwalk+r2v": Architecture.GAT,
    "GAT+node2vec+random": Architecture.GAT,
    "GAT+node2vec+r2v": Architecture.GAT,
    "deepwalk": Architecture.Word2Vec,
    "node2vec": Architecture.Word2Vec,
    "residual2vec": Architecture.Word2Vec,
    "groupbiased+residual2vec": Architecture.Word2Vec,
    "baseline+deepwalk": False,
    "baseline+node2vec": False,
    "groupbiased+gat+deepwalk": Architecture.GAT,
    "groupbiased+gat+node2vec": Architecture.GAT,
    "groupbiased+gcn+deepwalk": Architecture.GCN,
    "groupbiased+gcn+node2vec": Architecture.GCN,
}

EMB_MANIPULATION_METHODS = {
    "fairwalk+deepwalk": True,
    "fairwalk+node2vec": True,
    "crosswalk+deepwalk": True,
    "crosswalk+node2vec": True,
    "GCN+deepwalk+random": False,
    "GCN+deepwalk+r2v": True,
    "GCN+node2vec+random": False,
    "GCN+node2vec+r2v": True,
    "GAT+deepwalk+random": False,
    "GAT+deepwalk+r2v": True,
    "GAT+node2vec+random": False,
    "GAT+node2vec+r2v": True,
    "deepwalk": False,
    "node2vec": False,
    "residual2vec": True,
    "groupbiased+residual2vec": True,
    "baseline+deepwalk": True,
    "baseline+node2vec": True,
    "groupbiased+gat+deepwalk": True,
    "groupbiased+gat+node2vec": True,
    "groupbiased+gcn+deepwalk": True,
    "groupbiased+gcn+node2vec": True,
}

EMB_MANIPULATION_RENAME = {
    "fairwalk+deepwalk": "Fairwalk",
    "fairwalk+node2vec": "Fairwalk",
    "crosswalk+deepwalk": "Crosswalk",
    "crosswalk+node2vec": "Crosswalk",
    "GCN+deepwalk+r2v": "GCN (debiased)",
    "GCN+node2vec+r2v": "GCN (debiased)",
    "GAT+deepwalk+r2v": "GAT (debiased)",
    "GAT+node2vec+r2v": "GAT (debiased)",
    "residual2vec": "Deepwalk (debiased)",
    "baseline+deepwalk": "Bolkubasi",
    "baseline+node2vec": "Bolkubasi",
    "groupbiased+residual2vec": "Deepwalk (debiased)",
    "groupbiased+gat+deepwalk": "GAT (debiased)",
    "groupbiased+gat+node2vec": "GAT (debiased)",
    "groupbiased+gcn+deepwalk": "GCN (debiased)",
    "groupbiased+gcn+node2vec": "GCN (debiased)",
}

bcmap = sns.color_palette("Set3").as_hex()
ARCHITECTURE_TO_COLOR = {
    "GCN": bcmap[3],
    "GAT": bcmap[5],
    "word2vec": bcmap[7],
    "fairwalk": bcmap[4],
    "crosswalk": bcmap[2],
}

EMB_MANIPULATION_MODEL_TO_COLOR = {
    # very cool
    "fairwalk+deepwalk": bcmap[4],
    "fairwalk+node2vec": bcmap[4],
    "crosswalk+deepwalk": bcmap[2],
    "crosswalk+node2vec": bcmap[2],
    "GCN+deepwalk+r2v": bcmap[3],
    "GCN+node2vec+r2v": bcmap[3],
    "GAT+deepwalk+r2v": bcmap[5],
    "GAT+node2vec+r2v": bcmap[5],
    "residual2vec": bcmap[7],
    "baseline+deepwalk": bcmap[11],
    "baseline+node2vec": bcmap[11],
    "groupbiased+residual2vec": bcmap[7],
    "groupbiased+gat+deepwalk": bcmap[5],
    "groupbiased+gat+node2vec": bcmap[5],
    "groupbiased+gcn+deepwalk": bcmap[3],
    "groupbiased+gcn+node2vec": bcmap[3],
}

reference = {
    "DeepWalk": "s",
    "GCN": "o",
    "GAT": "d",
    "Fairwalk": "v",
    "CrossWalk": "^",
    "Manipulation of embedding": "p",
}

EMB_MANIPULATION_MODEL_TO_MARKER = {
    "fairwalk+deepwalk": reference["Fairwalk"],
    "fairwalk+node2vec": reference["Fairwalk"],
    "crosswalk+deepwalk": reference["CrossWalk"],
    "crosswalk+node2vec": reference["CrossWalk"],
    "GCN+deepwalk+r2v": reference["GCN"],
    "GCN+node2vec+r2v": reference["GCN"],
    "GAT+deepwalk+r2v": reference["GAT"],
    "GAT+node2vec+r2v": reference["GAT"],
    "residual2vec": reference["DeepWalk"],
    "baseline+deepwalk": reference["Manipulation of embedding"],
    "baseline+node2vec": reference["Manipulation of embedding"],
    "groupbiased+residual2vec": reference["DeepWalk"],
    "groupbiased+gat+deepwalk": reference["GAT"],
    "groupbiased+gat+node2vec": reference["GAT"],
    "groupbiased+gcn+deepwalk": reference["GCN"],
    "groupbiased+gcn+node2vec": reference["GCN"],
}

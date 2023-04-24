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
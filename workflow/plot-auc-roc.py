# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-01-17 08:52:11
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-01-18 01:49:00
# %%
import numpy as np
import pandas as pd
from scipy import sparse
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_file = snakemake.output["output_file"]
else:
    input_file = "../data/derived/results/result_data_all.csv"
    focal_model_list = [
        "fairwalk+deepwalk",
        "crosswalk+deepwalk",
        "deepwalk",
        "word2vec",
        "GCN+deepwalk+random",
        "GCN+deepwalk+r2v",
        "GAT+deepwalk+random",
        "GAT+deepwalk+r2v",
    ]
    output_file = "../data/"

# ========================
# Load
# ========================
data_table = pd.read_csv(input_file)

# %%
# ========================
# Preprocess
# ========================
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
    "word2vec": "DeepWalk (debiased)",  # What's this? residual2vec?
    "word2vec+deepwalk+random": "word2vec-deepwalk???",  # What's this?
}

model2group = {
    "fairwalk+deepwalk": "Fairwalk",
    "fairwalk+node2vec": "Fairwalk-node2vec",
    "crosswalk+deepwalk": "CrossWalk",
    "crosswalk+node2vec": "CrossWalk-node2vec",
    "GCN+deepwalk+random": "GCN",
    "GCN+deepwalk+r2v": "GCN",
    "GCN+node2vec+random": "GCN-node2vec",
    "GCN+node2vec+r2v": "GCN-node2vec",
    "GAT+deepwalk+random": "GAT",
    "GAT+deepwalk+r2v": "GAT",
    "GAT+node2vec+random": "GAT-node2vec",
    "GAT+node2vec+r2v": "GAT-node2vec",
    "deepwalk": "DeepWalk",
    "node2vec": "node2vec",
    "word2vec": "DeepWalk",  # What's this? residual2vec?
    "word2vec+deepwalk+random": "word2vec-deepwalk???",  # What's this?
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
    "word2vec": "Debiased",  # What's this? residual2vec?
    "word2vec+deepwalk+random": "Uniform??",  # What's this?
}

data_order = ["polbook", "polblog", "airport", "pokec"]

plot_data = data_table.copy()

# Filtering
plot_data = plot_data[plot_data["model"].isin(focal_model_list)]
plot_data = plot_data[plot_data["edgeSampling"].isin(["degree-group-biased"])]

# Append new columns
plot_data["modelType"] = plot_data["model"].map(model2type)
plot_data["Model"] = plot_data["model"].map(model2group)
plot_data["model"] = plot_data["model"].map(model_names)
plot_data = plot_data.rename(columns={"score": "AUC-ROC"})
# %%
# ========================
# Color and Style
# ========================
model_order = [
    "DeepWalk",  # Original
    "Fairwalk",  # Baseline #1
    "CrossWalk",  # Baseline #2,
    "DeepWalk (debiased)",  # residual2vec,
    "GCN",  # Baseline #3
    "GCN (debiased)",  # GCN + proposed
    "GAT",  # Baseline #4,
    "GAT (debiased)",  # residual2vec,
]

mcmap = sns.color_palette().as_hex()
bcmap = sns.color_palette("bright").as_hex()
cmap = sns.color_palette("colorblind").as_hex()

model_colors = {
    "GCN": cmap[0],
    "GAT": cmap[1],
    "DeepWalk": cmap[3],
    "Fairwalk": "#2d2d2d",
    "CrossWalk": "#8d8d8d",
}

#
# ========================
# Plot
# ========================
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

g = sns.catplot(
    data=plot_data,
    y="AUC-ROC",
    x="modelType",
    col="data",
    col_order=data_order,
    hue="Model",
    order=["Vanilla", "Debiased"],
    palette=model_colors,
    hue_order=["DeepWalk", "GCN", "GAT", "Fairwalk", "CrossWalk"],
    markers=["s", "o", "d", "v", "^"],
    linestyles=["-", "--", ":", "-.", "-"],
    color="k",
    kind="point",
    height=4,
    aspect=0.6,
)
g.set(ylim=(None, 1))

for ax in g.axes.flat:
    ax.axhline(0.5, ls=":", color="k")
g.set_xlabels("")

# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-01-17 08:52:11
# @Last Modified by:   Ashutosh Tiwari
# @Last Modified time: 2023-02-13 20:15:44
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
    focal_model_list = snakemake.params["focal_model_list"]
else:
    input_file = "../data/derived/results/result_disparity.csv"
    focal_model_list = [
        "fairwalk+deepwalk",
        "crosswalk+deepwalk",
        "deepwalk",
        "residual2vec",
        "GCN+deepwalk+random",
        "GCN+deepwalk+r2v",
        "GAT+deepwalk+random",
        "GAT+deepwalk+r2v",
        "baseline+node2vec",
        # "baseline"
    ]
    output_file = "../data/rank.png"
# ========================
# Load
# ========================
data_table = pd.read_csv(input_file)

# %%
# ========================
# Preprocess
# ========================

from model_styles import (
    model_names,
    model2group,
    model2type,
    model2markers,
    model2linestyle,
    model_order,
    model2colors,
)

markers = [model2markers[k] for k in model_order]
linestyles = [model2linestyle[k] for k in model_order]
cmap = sns.color_palette().as_hex()
bcmap = sns.color_palette("bright").as_hex()
mcmap = sns.color_palette("colorblind").as_hex()

neural_emb_color = "red"
spec_emb_color = mcmap[0]
com_color = mcmap[5]
neural_emb_color_2 = mcmap[2]
model2color = {
    "DeepWalk (debiased)": neural_emb_color,
    "GCN-node2vec (debiased)": sns.desaturate(neural_emb_color, 0.8),
    "GAT-node2vec (debiased)": sns.desaturate(neural_emb_color, 0.4),
    "node2vec": spec_emb_color,
    "GCN-node2vec": sns.desaturate(spec_emb_color, 0.4),
    "GAT-node2vec": sns.desaturate(spec_emb_color, 0.1),
    "Fairwalk (node2vec)": "#2d2d2d",
    "CrossWalk (node2vec)": "#8d8d8d",
    "Baseline-node2vec (debiased)": "red"
}
model2marker = {
    "DeepWalk (debiased)": "o",
    "GCN-node2vec (debiased)": "s",
    "GAT-node2vec (debiased)": "D",
    "node2vec": "o",
    "GCN-node2vec": "s",
    "GAT-node2vec": "D",
    "Fairwalk (node2vec)": "v",
    "CrossWalk (node2vec)": "^",
    "Baseline-node2vec (debiased)": "p",
}

data_order = ["polbook", "polblog", "airport", "twitch", "facebook"]

plot_data = data_table.copy()

# Filtering
plot_data = plot_data[plot_data["model"].isin(focal_model_list)]

# Append new columns
plot_data["modelType"] = plot_data["model"].map(model2type)
plot_data["Model"] = plot_data["model"].map(model2group)
plot_data["model"] = plot_data["model"].map(model_names)
plot_data = plot_data.rename(columns={"relativeEntropy": "Disparity"})

#
# ========================
# Plot
# ========================
sns.set_style("white")
sns.set(font_scale=1.1)
sns.set_style("ticks")

# fig, ax = plt.subplots(figsize=(5, 5))
g = sns.FacetGrid(data=plot_data, col="data", sharey=False, sharex=False, height=4)

for i, ax in enumerate(g.axes.flat):

    ax = sns.lineplot(
        data=plot_data[plot_data["data"] == data_order[i]],
        x="k",
        y="Disparity",
        hue="model",
        style="model",
        palette=model2color,
        markersize=10,
        markers=model2marker,
        hue_order=[
            "DeepWalk (debiased)",
            "GCN-node2vec (debiased)",
            "GAT-node2vec (debiased)",
            "node2vec",
            "GCN-node2vec",
            "GAT-node2vec",
            "Fairwalk (node2vec)",
            "CrossWalk (node2vec)",
            "Baseline-node2vec (debiased)"
        ][::-1],
        ax=ax,
    )
    ax.legend().remove()
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_title(data_order[i], fontsize=18)
g.axes.flat[0].set_ylabel("Disparity", fontsize=18)
g.fig.text(0.5, 0, "Rank", va="top", fontsize=18)

(dummy,) = ax.plot(
    [100],
    [0.01],
    marker="None",
    linestyle="None",
    label="dummy-tophead",
)

current_handles, current_labels = ax.get_legend_handles_labels()
new_handles = []
new_labels = []

model_group = {model_names[k]: v for k, v in model2type.items()}
prev_group = current_labels[0]
for i, l in enumerate(current_labels):
    if l not in model_group:
        continue

    curr_group = model_group[l]
    if prev_group != curr_group:
        new_handles.append(dummy)
        new_labels.append("")
    new_handles.append(current_handles[i])
    new_labels.append(model_names[l] if l in model_names else l)
    prev_group = curr_group

lgd = ax.legend(
    new_handles[::-1],
    new_labels[::-1],
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(1, 1),
    ncol=1,
    fontsize=11.5,
)
sns.despine()

ax.set_xscale("log")

print(output_file)
g.fig.savefig(output_file, bbox_inches="tight", dpi=300)
# %%

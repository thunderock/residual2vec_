# -*- coding: utf-8 -*-
# @Author: Ashutosh Tiwari
# @Email: checkashu@gmail.com
# @Date:   2023-04-22 20:14:43
# @Filepath: workflow/plot_auc_roc.py
# %%
import os
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D as ln
import matplotlib as mpl
import seaborn as sns
from matplotlib.patheffects import withStroke
import matplotlib.ticker as ticker

INPUT_FILE = "../data/derived/results/result_auc_roc.csv"
OUTPUT_PROPOSED_VS_BASELINE = "../figs/auc_roc_proposed_vs_baseline.png"
OUTPUT_MANIPULATION_METHODS = "../figs/auc_roc_manipulation_methods.png"
DATASETS = ["polbook", "polblog", "airport", "twitch", "facebook"]

FOCAL_MODEL_LIST = [
    # order this in the order of complexity
    "groupbiased+residual2vec",
    "groupbiased+gcn+deepwalk",
    "groupbiased+gat+deepwalk",
    "fairwalk+deepwalk",
    "crosswalk+deepwalk",
    "deepwalk",
    "GCN+deepwalk+random",
    # "GCN+deepwalk+r2v",
    "GAT+deepwalk+random",
    # "GAT+deepwalk+r2v",
    "baseline+deepwalk",  # replace this with baseline + deepwalk
]
SAMPLING_METHOD = "uniform"
DATASETS = ["polbook", "polblog", "airport", "twitch", "facebook"]

if "snakemake" in sys.modules:
    INPUT_FILE = snakemake.input["input_file"]
    OUTPUT_PROPOSED_VS_BASELINE = snakemake.output[0]
    OUTPUT_MANIPULATION_METHODS = snakemake.output[1]
    FOCAL_MODEL_LIST = snakemake.params["focal_model_list"]
    SAMPLING_METHOD = snakemake.params["sampling_method"]

print("INPUT_FILE: ", INPUT_FILE)
print("OUTPUT_PROPOSED_VS_BASELINE: ", OUTPUT_PROPOSED_VS_BASELINE)
print("OUTPUT_MANIPULATION_METHODS: ", OUTPUT_MANIPULATION_METHODS)
print("FOCAL_MODEL_LIST: ", FOCAL_MODEL_LIST)
print("SAMPLING_METHOD: ", SAMPLING_METHOD)


df = pd.read_csv(INPUT_FILE)
df = df[df.edgeSampling == SAMPLING_METHOD]
df = df[["score", "data", "sampleId", "model"]]

from model_styles import (
    model_names,
    model2group,
    model2type,
    model2markers,
    model2linestyle,
    model2colors,
)
from new_model_styles import (
    MODEL_TO_IS_ARCHITECTURE,
    EMB_MANIPULATION_METHODS,
    EMB_MANIPULATION_RENAME,
    EMB_MANIPULATION_MODEL_TO_COLOR,
    EMB_MANIPULATION_MODEL_TO_MARKER,
)

model2arch = model2group.copy()
model2arch["fairwalk+deepwalk"] = "word2vec"
model2arch["fairwalk+node2vec"] = "word2vec"
model2arch["crosswalk+deepwalk"] = "word2vec"
model2arch["crosswalk+node2vec"] = "word2vec"

df["type"] = df["model"].map(model2type)
df["architecture"] = df["model"].map(model2arch)
df["data"] = df["data"].map(lambda x: x.capitalize())
# %%
# First figure
#
baseline_models = ["fairwalk+deepwalk", "crosswalk+deepwalk"]

focal_model = [
    model for model in FOCAL_MODEL_LIST if MODEL_TO_IS_ARCHITECTURE[model]
] + baseline_models
plot_data = df[df.model.isin(focal_model)]


plot_data["type_model"] = plot_data.apply(
    lambda x: EMB_MANIPULATION_RENAME[x["model"]]
    if x["model"] in baseline_models
    else x["type"] + "+" + x["architecture"],
    axis=1,
)
plot_data["type_model"].unique()
# %%
sns.set_style("white")
sns.set(font_scale=1)
sns.set_style("ticks")

#
# Main plot
#
# Plot the scatter plot
cmap = sns.color_palette("Set3").as_hex()
cmap2 = sns.color_palette("bright").as_hex()
arch2color = {
    "Fairwalk": cmap[0],
    "Crosswalk": cmap[2],
    "Debiased+GCN": cmap2[3],
    "Debiased+GAT": cmap2[3],
    "Debiased+word2vec": cmap2[3],
}
palette = {
    k: "#efefef" if "Vanilla" in k else arch2color[k]
    for k in plot_data["type_model"].unique()
}
# palette = {"Vanilla": "#adadad", "Debiased": sns.color_palette().as_hex()[3]}

g = sns.catplot(
    col="data",
    x="architecture",
    y="score",
    hue="type_model",
    data=plot_data,
    order=["word2vec", "GCN", "GAT"],
    # capsize=0.15,
    # join=False,
    # kind="point",
    palette=palette,
    sharex=True,
    sharey=False,
    kind="strip",
    height=3.5,
    aspect=0.8,
    # scale=0.5,
    # errwidth=1.5,
    # errorbar=("ci", 95),
    legend=True,
    edgecolor="k",
    size=6.5,
    linewidth=0.8,
    dodge=True,
    native_scale=True,
    jitter=0.3,
    legend_out=True,
)
# Plot the scatter boxplot
#
# Styling
#
g.set_axis_labels("", "AUC-ROC")
g.fig.text(0.5, 0.02, "Model architecture")

g.set_titles(col_template="{col_name}")
g.set(xlim=(-0.5, None))

for i, arch in enumerate(range(3)):
    for ax in g.axes.flat:
        ax.axvline(i + 0.5, color="k", lw=0.5, ls=":")
        # ax.axvspan(-0.5 + i, 0.5 + i, facecolor=arch2color[arch], alpha=0.1, zorder=-1)
sns.despine()
plt.savefig(OUTPUT_PROPOSED_VS_BASELINE, dpi=300, bbox_inches="tight")
plt.close()

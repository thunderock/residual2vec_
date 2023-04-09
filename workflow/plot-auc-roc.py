# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-01-17 08:52:11
# @Last Modified by:   Ashutosh Tiwari
# @Last Modified time: 2023-04-09 17:48:41
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
    print("output_file: ", output_file)
    focal_model_list = snakemake.params["focal_model_list"]
else:
    input_file = "../data/derived/results/result_auc_roc.csv"
    focal_model_list = [
        "fairwalk+node2vec",
            "crosswalk+node2vec",
            "GCN+node2vec+random",
            "GCN+node2vec+r2v",
            "GAT+node2vec+random",
            "GAT+node2vec+r2v",
    ]
    output_file = "/tmp/roc.png"

print(focal_model_list)
# ========================
# Load
# ========================
data_table = pd.read_csv(input_file)

# %%
# ========================
# Preprocess
# ========================

from model_styles import model_names, model2group, model2type, model2markers, model2linestyle, model_order, model2colors

MODEL_ORDER = model_order if 'deepwalk' in focal_model_list else model_order[1:]

markers = [model2markers[k] for k in MODEL_ORDER]
linestyles = [model2linestyle[k] for k in MODEL_ORDER]

data_order = ["polbook", "polblog", "airport", "twitch", "facebook"]


plot_data = data_table.copy()

# Filtering
plot_data = plot_data[plot_data["model"].isin(focal_model_list)]
plot_data = plot_data[plot_data["edgeSampling"].isin(["uniform"])]

# Append new columns
plot_data["modelType"] = plot_data["model"].map(model2type)
plot_data["Model"] = plot_data["model"].map(model2group)
plot_data["model"] = plot_data["model"].map(model_names)
plot_data = plot_data.rename(columns={"score": "AUC-ROC"})

#
# ========================
# Plot
# ========================
sns.set_style("white")
sns.set(font_scale=1.1)
sns.set_style("ticks")

g = sns.catplot(
    data=plot_data,
    y="AUC-ROC",
    x="modelType",
    col="data",
    col_order=data_order,
    hue="Model",
    order=["Vanilla", "Debiased"],
    palette=model2colors,
    hue_order=MODEL_ORDER,
    markers=markers,
    linestyles=linestyles,
    color="k",
    kind="point",
    height=3,
    aspect=0.8,
    sharey=False
)
g.set(ylim=(None, 1))

for ax in g.axes.flat:
    ax.axhline(0.5, ls=":", color="k")
g.set_xlabels("")
sns.move_legend(g, "lower center", bbox_to_anchor=(.45, -0.005), ncol=len(MODEL_ORDER), fontsize=8)
g.legend.set_title(None)
g.fig.savefig(output_file, bbox_inches='tight', dpi=300)

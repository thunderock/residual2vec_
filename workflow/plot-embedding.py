# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-01-24 00:52:33
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-02-14 10:41:49
# %%
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import numpy as np
import pandas as pd
from scipy import sparse
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

if "snakemake" in sys.modules:
    emb_file = snakemake.params["emb_file"]
    node_table_file = snakemake.input["node_table_file"]
    output_file = snakemake.output["output_file"]
else:
    emb_file = "../data/derived/preprocessed/polbook/polbook_three/polbook_word2vec_None_deepwalk_deepwalk_embs.npy"
    node_table_file = "../data/derived/preprocessed/polbook/node_table.csv"

# ========================
# Load
# ========================
emb = np.load(emb_file)
node_table = pd.read_csv(node_table_file)

is_polbook_data = "polbook" in emb_file  # for figure 1 in the main text

# %%========================
# Preprocessing
# ========================
# Get the group membership
group_labels = node_table["group_id"].values
group_ids = np.unique(group_labels, return_inverse=True)[1]

# Number of groups
K = int(np.max(group_ids) + 1)

# Number of nodes
N = emb.shape[0]

# ========================
# PCA Projection
# ========================
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor

clf = LocalOutlierFactor(n_neighbors=40, contamination=0.1)
s = clf.fit_predict(emb)
xy = PCA(n_components=2, whiten=True).fit(emb[s > 0, :]).transform(emb)

# ========================
# Styles
# ========================

if is_polbook_data:
    plot_data = pd.DataFrame(
        {
            "x": xy[:, 0],
            "y": xy[:, 1],
            "Group": np.array(["Liberal", "Conservative", "Neutral"])[group_ids],
        }
    )
    cmap = sns.color_palette("tab10")
    colors = {"Liberal": cmap[0], "Conservative": cmap[3], "Neutral": cmap[4]}
else:
    plot_data = pd.DataFrame(
        {
            "x": xy[:, 0],
            "y": xy[:, 1],
            "Group": group_labels,
        }
    )
    cmap = sns.color_palette().as_hex()
    colors = {k: cmap[i] for i, k in enumerate(np.unique(group_labels))}
hue_order = np.sort(list(colors.keys()))
# ========================
# Plot
# ========================
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
if is_polbook_data:
    figsize = (3, 3)
else:
    figsize = (12, 12)
fig, ax = plt.subplots(figsize=figsize)

g = sns.scatterplot(
    data=plot_data,
    x="x",
    y="y",
    hue="Group",
    palette=colors,
    edgecolor="k",
    hue_order=hue_order,
    s=80,
    ax=ax,
)
ax.legend(
    loc="upper left",
    bbox_to_anchor=(1, 1),
    frameon=False,
    ncols=3,
    handletextpad=0.1,
)
ax.axis("off")

fig.savefig("tmp.pdf", bbox_inches="tight", dpi=300)

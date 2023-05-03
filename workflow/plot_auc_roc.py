# -*- coding: utf-8 -*-
# @Author: Ashutosh Tiwari
# @Email: checkashu@gmail.com
# @Date:   2023-04-22 20:14:43
# @Filepath: workflow/plot_auc_roc.py
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

INPUT_FILE = "data/derived/results/result_auc_roc.csv"
OUTPUT_PROPOSED_VS_BASELINE = 'figs/auc_roc_proposed_vs_baseline.png'
OUTPUT_MANIPULATION_METHODS = 'figs/auc_roc_manipulation_methods.png'
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
            "baseline+deepwalk" # replace this with baseline + deepwalk
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
df = df[['score', 'data', 'sampleId', 'model']]

from model_styles import model_names, model2group, model2type, model2markers, model2linestyle, model2colors
from new_model_styles import MODEL_TO_IS_ARCHITECTURE, EMB_MANIPULATION_METHODS, EMB_MANIPULATION_RENAME, EMB_MANIPULATION_MODEL_TO_COLOR, EMB_MANIPULATION_MODEL_TO_MARKER

df['type'] = df['model'].map(model2type)
df['architecture'] = df['model'].map(model2group)
df['data'] = df['data'].map(lambda x: x.capitalize())

g = sns.catplot(col='data', y='score', hue='type', data=df[df.model.isin([model for model in FOCAL_MODEL_LIST if MODEL_TO_IS_ARCHITECTURE[model]])],
                capsize=.15, join=False, 
                x="architecture", kind="point", sharex=True, sharey=True,
        palette = {"Vanilla":"grey", "Debiased":sns.color_palette().as_hex()[3]}, height=4, aspect=.5, scale=0.5, errwidth=1.5, errorbar=('ci', 95), legend=False)
g.set_axis_labels("", "AUC-ROC")
for (i,j,k), data in g.facet_data():
    ax = g.facet_axis(i, j)
    if j > 0:
        ax.get_yaxis().set_visible(False)
        # remove y axis line
        ax.spines['left'].set_visible(False)
        # ax.get_yaxis().set_ticks([])
    else:
        ax.legend(loc="upper right", borderaxespad=0.)
    for child in ax.findobj(ln):
        child.set(path_effects=[withStroke(linewidth=1.5, foreground='black')])
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.despine()
plt.savefig(OUTPUT_PROPOSED_VS_BASELINE, dpi=300, bbox_inches='tight')
plt.close()


model_order = [model for model in FOCAL_MODEL_LIST if EMB_MANIPULATION_METHODS[model]]
# df.to_csv("/tmp/df.csv", index=False)

ax = sns.pointplot(x='data', y='score', hue='model', 
              data=df[df.model.isin(model_order)], 
              join=False, capsize=.15, palette=EMB_MANIPULATION_MODEL_TO_COLOR,
              markers=[EMB_MANIPULATION_MODEL_TO_MARKER[i] for i in model_order],
              hue_order=model_order,
              errwidth=1.5, scale=0.5, errorbar=('ci',95), 
              )
legend_handles, _= ax.get_legend_handles_labels()

separator = 0
for model in model_order:
    if 'debiased' not in EMB_MANIPULATION_RENAME[model]:
        break
    separator += 1
    
legend_renames = [EMB_MANIPULATION_RENAME[i.get_label()] for i in legend_handles]

legend_handles.insert(separator, mpl.lines.Line2D([], [], linestyle=''))
legend_renames.insert(separator, "")
# inside the plot
ax.legend(legend_handles, legend_renames, prop={'size': 6}, loc='lower right')

# bring ticks closer
ax.tick_params(axis='x', which='major', pad=0.5)
for child in ax.findobj(ln):
        child.set(path_effects=[withStroke(linewidth=1.5, foreground='black')])

plt.title("Comparison of debiased models")
plt.xlabel("Dataset")
plt.ylabel("AUC-ROC")
sns.despine()
plt.savefig(OUTPUT_MANIPULATION_METHODS, dpi=300, bbox_inches='tight')
plt.close()


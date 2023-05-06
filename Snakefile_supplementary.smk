import numpy as np
from os.path import join as j
import itertools
import pandas as pd
from snakemake.utils import Paramspace


# =====================
# Output
# =====================
FIG_EMB = j("figs", "embedding", "data~{data}_sampleId~{sampleId}_model~{model}.png")
MODEL_LIST = [
    "fairwalk+deepwalk",
    "fairwalk+node2vec",
    "crosswalk+deepwalk",
    "crosswalk+node2vec",
    "GCN+deepwalk+random",
    "GCN+deepwalk+r2v",
    "GCN+node2vec+random",
    "GCN+node2vec+r2v",
    "GAT+deepwalk+random",
    "GAT+deepwalk+r2v",
    "GAT+node2vec+random",
    "GAT+node2vec+r2v",
    "deepwalk",
    "node2vec",
    "residual2vec",
    "groupbiased+residual2vec",
    "baseline+deepwalk",
    "baseline+node2vec",
    "groupbiased+gat+deepwalk",
    "groupbiased+gat+node2vec",
    "groupbiased+gcn+deepwalk",
    "groupbiased+gcn+node2vec",
]
SRC_DATA_ROOT = '../final_/'
NODE_TABLE_FILE = j(SRC_DATA_ROOT, "{data}/node_table.csv")
MODEL2EMBFILE_POSTFIX= {
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
    "groupbiased+gcn+node2vec": "_gcn_node2vec_r2v_groupbiased_embs.npy",
    }

# =====================
# Main output
# =====================
rule supplementary_figs:
    input:
        expand(FIG_EMB, data = ["polbook"], sampleId=["two"], model=MODEL_LIST),

# =====================
# Plot
# =====================
rule plot_embedding:
    input:
        node_table_file = NODE_TABLE_FILE,
    params:
        emb_file = lambda wildcards: "{root}/{data}/{data}_{sampleId}/{data}".format(root=SRC_DATA_ROOT, data=wildcards.data, sampleId=wildcards.sampleId)+MODEL2EMBFILE_POSTFIX[wildcards.model] # not ideal but since the file names are different, I generate the file name in the script and load the corresponding file.
    output:
        output_file = FIG_EMB
    script:
        "workflow/plot-embedding.py"
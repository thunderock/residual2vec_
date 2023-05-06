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
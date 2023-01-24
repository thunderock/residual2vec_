import numpy as np
from os.path import join as j
import itertools
import pandas as pd
from snakemake.utils import Paramspace
include: "./utils/workflow_utils.smk" # not able to merge this with snakemake_utils.py due to some path breakage issues

# =====================
# Output
# =====================
FIG_EMB = j("figs", "embedding", "data~{data}_sampleId~{sampleId}_model~{model}.pdf")

# =====================
# Main output
# =====================
rule supplementary_figs:
    input:
        expand(FIG_EMB, data = ["polbook"], sampleId=["one"], model=MODEL_LIST),

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
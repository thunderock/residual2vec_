import numpy as np
from os.path import join as j
import itertools
import pandas as pd
from snakemake.utils import Paramspace
# include: "./utils/workflow_utils.smk" # not able to merge this with snakemake_utils.py due to some path breakage issues

# ====================
# Root folder path setting
# ====================

# network file
SRC_DATA_ROOT = j("..", "final_128")
DERIVED_DIR = j("data", "derived")

DATA_LIST = ["airport", "polbook", "polblog", "pokec"]
SAMPLE_ID_LIST = ["one", "two", "three", "four", "five"] # why not arabic numbers?
N_ITERATION = 1

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
    "residual2vec"
]

MODEL2EMBFILE_POSTFIX= {
    "fairwalk+deepwalk": "_fairwalk_deepwalk.npy",
    "fairwalk+node2vec": "_fairwalk_node2vec.npy",
    "crosswalk+deepwalk": "_crosswalk_deepwalk.npy",
    "crosswalk+node2vec": "_crosswalk_node2vec.npy",
    "GCN+deepwalk+random": "_gcn_None_deepwalk_deepwalk_embs.npy",
    "GCN+deepwalk+r2v": "_gcn_None_deepwalk_r2v_embs.npy",
    "GCN+node2vec+random": "_gcn_None_node2vec_deepwalk_embs.npy",
    "GCN+node2vec+r2v": "_gcn_None_node2vec_r2v_embs.npy",
    "GAT+deepwalk+random": "_gat_None_deepwalk_deepwalk_embs.npy",
    "GAT+deepwalk+r2v": "_gat_None_deepwalk_r2v_embs.npy",
    "GAT+node2vec+random": "_gat_None_node2vec_deepwalk_embs.npy",
    "GAT+node2vec+r2v": "_gat_None_node2vec_r2v_embs.npy",
    "deepwalk": "_deepwalk_128_embs.npy",
    "node2vec": "_node2vec_128_embs.npy",
    "residual2vec": "_residual2vec_128_embs.npy"
}

# ====================
# Input files
# ====================

TRAIN_NET_FILE = j(SRC_DATA_ROOT, "{data}/{data}_{sampleId}/{data}_adj.npz") # train
TEST_NET_FILE = j(SRC_DATA_ROOT, "{data}/{data}_{sampleId}/{data}_test_adj.npz") # test
NODE_TABLE_FILE = j(SRC_DATA_ROOT, "{data}/node_table.csv") # ndoe table

# ====================
# Output files
# ====================
RESULT_DIR = j(DERIVED_DIR, "results")

LP_DATASET_DIR = j(DERIVED_DIR, "link-prediction-dataset")
LP_DATASET_FILE = j(LP_DATASET_DIR, "data~{data}_edgeSampling~{edgeSampling}_sampleId~{sampleId}_iteration~{iteration}.csv")

# AUC-ROC
LP_SCORE_FILE = j(RESULT_DIR, "auc_roc", "result_data~{data}_edgeSampling~{edgeSampling}_sampleId~{sampleId}_model~{model}_iteration~{iteration}.csv")
LP_ALL_SCORE_FILE = j(RESULT_DIR, "result_auc_roc.csv")

# Disparity score
DISPARITY_SCORE_FILE = j(RESULT_DIR, "disparity", "result_data~{data}_sampleId~{sampleId}_model~{model}.csv")
DISPARITY_ALL_SCORE_FILE = j(RESULT_DIR, "result_disparity.csv")

#
# Figures
#
FIG_LP_SCORE_DEEPWALK = j("figs", "aucroc_deepwalk.pdf")
FIG_DISPARITY_SCORE_DEEPWALK= j("figs", "disparity_deepwalk.pdf")
FIG_DISPARITY_CURVE_DEEPWALK = j("figs", "disparity-curve_deepwalk.pdf")

FIG_LP_SCORE_NODE2VEC = j("figs", "aucroc_node2vec.pdf")
FIG_DISPARITY_SCORE_NODE2VEC= j("figs", "disparity_node2vec.pdf")
FIG_DISPARITY_CURVE_NODE2VEC = j("figs", "disparity-curve_node2vec.pdf")
# ===================
# Configurations
# ===================

lp_benchmark_params = {
    "edgeSampling":["uniform", "degree-biased", "degree-group-biased"],
    "model":MODEL_LIST,
    "sampleId":["one", "two", "three", "four", "five"],
    "iteration":list(range(N_ITERATION))
}

disparity_benchmark_params = {
    "model":MODEL_LIST,
    "sampleId":["one", "two", "three", "four", "five"],
}

# =====================
# Main output
# =====================
rule link_prediction_all:
    input:
        expand(LP_ALL_SCORE_FILE, data = DATA_LIST),
        expand(DISPARITY_ALL_SCORE_FILE, data = DATA_LIST),

rule link_prediction_figs:
    input:
        FIG_LP_SCORE_DEEPWALK,
        FIG_DISPARITY_SCORE_DEEPWALK,
        FIG_DISPARITY_SCORE_NODE2VEC,
        FIG_LP_SCORE_NODE2VEC,
        FIG_DISPARITY_CURVE_DEEPWALK,
        FIG_DISPARITY_CURVE_NODE2VEC


# =====================
# Network generation
# =====================
rule generate_link_prediction_dataset:
    input:
        train_net_file = TRAIN_NET_FILE,
        test_net_file = TEST_NET_FILE,
        node_table_file = NODE_TABLE_FILE,
    params:
        samplerName = lambda wildcards: wildcards.edgeSampling
    output:
        output_file = LP_DATASET_FILE
    script:
        "workflow/generate-link-prediction-dataset.py"


# =====================
# Evaluation
# =====================
rule eval_link_prediction:
    input:
        input_file = LP_DATASET_FILE,
    params:
        emb_file = lambda wildcards: "{root}/{data}/{data}_{sampleId}/{data}".format(root=SRC_DATA_ROOT, data=wildcards.data, sampleId=wildcards.sampleId)+MODEL2EMBFILE_POSTFIX[wildcards.model] # not ideal but since the file names are different, I generate the file name in the script and load the corresponding file.
    output:
        output_file = LP_SCORE_FILE
    script:
        "workflow/evaluate-lp-performance.py"

rule eval_disparity:
    input:
        node_table_file = NODE_TABLE_FILE,
    params:
        emb_file = lambda wildcards: "{root}/{data}/{data}_{sampleId}/{data}".format(root=SRC_DATA_ROOT, data=wildcards.data, sampleId=wildcards.sampleId)+MODEL2EMBFILE_POSTFIX[wildcards.model] # not ideal but since the file names are different, I generate the file name in the script and load the corresponding file.
    output:
        output_file = DISPARITY_SCORE_FILE
    script:
        "workflow/evaluate-disparity.py"


rule concatenate_results:
    input:
        input_file_list = expand(LP_SCORE_FILE, **lp_benchmark_params, data = DATA_LIST),
    output:
        output_file = LP_ALL_SCORE_FILE
    script:
        "workflow/concat-results.py"

rule concatenate_disparity_results:
    input:
        input_file_list = expand(DISPARITY_SCORE_FILE, **disparity_benchmark_params, data = DATA_LIST),
    output:
        output_file = DISPARITY_ALL_SCORE_FILE
    script:
        "workflow/concat-results.py"

# =====================
# Plot
# =====================
rule plot_auc_roc_score_deepwalk:
    input:
        input_file = LP_ALL_SCORE_FILE
    params:
        focal_model_list = [
            "fairwalk+deepwalk",
            "crosswalk+deepwalk",
            "deepwalk",
            "residual2vec",
            "GCN+deepwalk+random",
            "GCN+deepwalk+r2v",
            "GAT+deepwalk+random",
            "GAT+deepwalk+r2v",
        ]
    output:
        output_file = FIG_LP_SCORE_DEEPWALK
    script:
        "workflow/plot-auc-roc.py"

rule plot_auc_roc_score_node2vec:
    input:
        input_file = LP_ALL_SCORE_FILE
    params:
        focal_model_list = [
            "fairwalk+node2vec",
            "crosswalk+node2vec",
            "GCN+node2vec+random",
            "GCN+node2vec+r2v",
            "GAT+node2vec+random",
            "GAT+node2vec+r2v",
        ]
    output:
        output_file = FIG_LP_SCORE_NODE2VEC
    script:
        "workflow/plot-auc-roc.py"

rule plot_disparity:
    input:
        input_file = DISPARITY_ALL_SCORE_FILE
    params:
        focal_model_list = [
            "fairwalk+deepwalk",
            "crosswalk+deepwalk",
            "deepwalk",
            "residual2vec",
            "GCN+deepwalk+random",
            "GCN+deepwalk+r2v",
            "GAT+deepwalk+random",
            "GAT+deepwalk+r2v",
        ]
    output:
        output_file = FIG_DISPARITY_SCORE_DEEPWALK
    script:
        "workflow/plot-disparity.py"

rule plot_disparity_node2vec:
    input:
        input_file = DISPARITY_ALL_SCORE_FILE
    params:
        focal_model_list = [
            "fairwalk+node2vec",
            "crosswalk+node2vec",
            "GCN+node2vec+random",
            "GCN+node2vec+r2v",
            "GAT+node2vec+random",
            "GAT+node2vec+r2v",
        ]
    output:
        output_file = FIG_DISPARITY_SCORE_NODE2VEC
    script:
        "workflow/plot-disparity.py"

rule plot_disparity_curve_deepwalk:
    input:
        input_file = DISPARITY_ALL_SCORE_FILE
    params:
        focal_model_list = [
            "fairwalk+deepwalk",
            "crosswalk+deepwalk",
            "deepwalk",
            "residual2vec",
            "GCN+deepwalk+random",
            "GCN+deepwalk+r2v",
            "GAT+deepwalk+random",
            "GAT+deepwalk+r2v",
        ]
    output:
        output_file = FIG_DISPARITY_CURVE_DEEPWALK
    script:
        "workflow/plot-disparity-curve.py"

rule plot_disparity_curve_node2vec:
    input:
        input_file = DISPARITY_ALL_SCORE_FILE
    params:
        focal_model_list = [
            "fairwalk+node2vec",
            "crosswalk+node2vec",
            "GCN+node2vec+random",
            "GCN+node2vec+r2v",
            "GAT+node2vec+random",
            "GAT+node2vec+r2v",
        ]
    output:
        output_file = FIG_DISPARITY_CURVE_NODE2VEC
    script:
        "workflow/plot-disparity-curve-node2vec.py"
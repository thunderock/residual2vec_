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
SRC_DATA_ROOT = j("..", "final_")
DERIVED_DIR = j("data", "derived")

DATA_LIST = [ "polbook", "polblog", "airport", "twitch", "facebook"]


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
    "residual2vec",
    "groupbiased+residual2vec",
    "baseline+deepwalk",
    "baseline+node2vec",
    "groupbiased+gat+deepwalk",
    "groupbiased+gat+node2vec",
    "groupbiased+gcn+deepwalk",
    "groupbiased+gcn+node2vec",
]

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
FIG_LP_SCORE_DEEPWALK_PROPOSED_COMPARISON = j("figs", "aucroc_deepwalk_proposed_comparison.pdf")

FIG_DISPARITY_SCORE_DEEPWALK= j("figs", "disparity_deepwalk.pdf")
FIG_DISPARITY_CURVE_DEEPWALK = j("figs", "disparity-curve_deepwalk.pdf")

FIG_LP_SCORE_NODE2VEC = j("figs", "aucroc_node2vec.pdf")
FIG_DISPARITY_SCORE_NODE2VEC= j("figs", "disparity_node2vec.pdf")
FIG_DISPARITY_CURVE_NODE2VEC = j("figs", "disparity-curve_node2vec.pdf")

FIG_LOCAL_FAIRNESS_PER_NODE = j("figs", "deepwalk_disparity_per_node.png")
FIG_LOCAL_FAIRNESS_PER_NODE_CW = j("figs", "deepwalk_disparity_per_node_cw.png")
FIG_GLOBAL_FAIRNESS_PER_NODE = j("figs", "deepwalk_disparity_per_node_global.png")
FIG_GLOBAL_FAIRNESS_PER_NODE_CW = j("figs", "deepwalk_disparity_per_node_global_cw.png")
# ===================
# Configurations
# ===================

lp_benchmark_params = {
    "edgeSampling":["uniform", "degree-biased"],
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
        FIG_DISPARITY_CURVE_NODE2VEC,
        FIG_LOCAL_FAIRNESS_PER_NODE,
        FIG_LOCAL_FAIRNESS_PER_NODE_CW,
        FIG_GLOBAL_FAIRNESS_PER_NODE,
        FIG_GLOBAL_FAIRNESS_PER_NODE_CW,

rule fairness_per_node:
    params:
        embs_mapping = MODEL2EMBFILE_POSTFIX,
        datasets = DATA_LIST,
        base_dir = SRC_DATA_ROOT,
        sample_ids = SAMPLE_ID_LIST
    output:
        FIG_LOCAL_FAIRNESS_PER_NODE,
        FIG_LOCAL_FAIRNESS_PER_NODE_CW,
    threads: 1
    script:
        "workflow/plot_fairness_per_node.py"

rule plot_global_fairness:
    params:
        embs_mapping = MODEL2EMBFILE_POSTFIX,
        datasets = DATA_LIST,
        base_dir = SRC_DATA_ROOT,
        sample_ids = SAMPLE_ID_LIST
    output:
        FIG_GLOBAL_FAIRNESS_PER_NODE,
        FIG_GLOBAL_FAIRNESS_PER_NODE_CW,
    threads: 1
    script:
        "workflow/plot_global_fairness.py"
        
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
# rule plot_auc_roc_score_deepwalk:
#     input:
#         input_file = LP_ALL_SCORE_FILE
#     params:
#         focal_model_list = [
#             "fairwalk+deepwalk",
#             "crosswalk+deepwalk",
#             "deepwalk",
#             "groupbiased+residual2vec",
#             "GCN+deepwalk+random",
#             "groupbiased+gcn+deepwalk",
#             # "GCN+deepwalk+r2v",
#             "GAT+deepwalk+random",
#             "groupbiased+gat+deepwalk",
#             # "GAT+deepwalk+r2v",
#             "baseline+deepwalk"
#         ]
#     output:
#         output_file = FIG_LP_SCORE_DEEPWALK
#     script:
#         "workflow/plot-auc-roc.py"

rule plot_auc_roc_score_deepwalk:
    input:
        input_file = LP_ALL_SCORE_FILE
    params:
        focal_model_list = [
            "fairwalk+deepwalk",
            "crosswalk+deepwalk",
            "deepwalk",
            "groupbiased+residual2vec",
            "GCN+deepwalk+random",
            "groupbiased+gcn+deepwalk",
            # "GCN+deepwalk+r2v",
            "GAT+deepwalk+random",
            "groupbiased+gat+deepwalk",
            # "GAT+deepwalk+r2v",
            "baseline+deepwalk" # replace this with baseline + deepwalk
        ],
        sampling_method = "uniform"
    output:
        FIG_LP_SCORE_DEEPWALK, FIG_LP_SCORE_DEEPWALK_PROPOSED_COMPARISON
    script:
        "workflow/plot_auc_roc.py"


rule plot_auc_roc_score_node2vec:
    input:
        input_file = LP_ALL_SCORE_FILE
    params:
        focal_model_list = [
            "fairwalk+node2vec",
            "crosswalk+node2vec",
            "GCN+node2vec+random",
            "groupbiased+gcn+node2vec",
            # "GCN+node2vec+r2v",
            "GAT+node2vec+random",
            "groupbiased+gat+node2vec",
            # "GAT+node2vec+r2v",
            "baseline+node2vec" # replace this with baseline + node2vec
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
            "groupbiased+residual2vec",
            "GCN+deepwalk+random",
            "groupbiased+gcn+deepwalk",
            # "GCN+deepwalk+r2v",
            "GAT+deepwalk+random",
            "groupbiased+gat+deepwalk",
            # "GAT+deepwalk+r2v",
            "baseline+deepwalk"
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
            "groupbiased+gcn+node2vec",
            
            # "GCN+node2vec+r2v",
            "GAT+node2vec+random",
            "groupbiased+gat+node2vec",
            # "GAT+node2vec+r2v",
            "baseline+node2vec"
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
            "groupbiased+residual2vec",
            "GCN+deepwalk+random",
            "groupbiased+gcn+deepwalk",
            # "GCN+deepwalk+r2v",
            "GAT+deepwalk+random",
            "groupbiased+gat+deepwalk",
            # "GAT+deepwalk+r2v",
            "baseline+deepwalk"
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
            "groupbiased+gcn+node2vec",
            # "GCN+node2vec+r2v",
            "GAT+node2vec+random",
            "groupbiased+gat+node2vec",
            # "GAT+node2vec+r2v",
            "baseline+node2vec"
        ]
    output:
        output_file = FIG_DISPARITY_CURVE_NODE2VEC
    script:
        "workflow/plot-disparity-curve-node2vec.py"

# =====================
# TODO: merge with the following snakemake
# =====================
include: "Snakefile_supplementary.smk"

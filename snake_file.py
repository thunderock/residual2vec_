import os
from os.path import join as j
from utils.snakemake_utils import get_string_boolean, FileResources
# os.environ["CUDA_VISIBLE_DEVICES"]=""
config = {'gnn_model': 'gat', 'crosswalk': 'false',
          'r2v': 'true', 'dataset': 'pokec', 'device': 'cpu'}



# ENV = config.get('env', 'remote')
# Variables in snakefile
GNN_MODEL = config.get('gnn_model', "gat")

# not in use right now
DATASET = config.get('dataset', 'pokec')

CROSSWALK = config.get('crosswalk', 'false')
CROSSWALK = get_string_boolean(CROSSWALK)

R2V = config.get('r2v', 'false')
R2V = get_string_boolean(R2V)

SET_DEVICE = config.get('device', 'cuda:0')
# DATA_ROOT = "/data/sg/ashutiwa/residual2vec_"
# if ENV in ('local', 'carbonate'):
DATA_ROOT = "data"


# variables sanity check
assert GNN_MODEL in ('gat', 'gcn')
assert DATASET in ('pokec', 'small_pokec', 'airport', 'polblog', 'polbook')
assert CROSSWALK in (True, False)
assert R2V in (True, False)
assert SET_DEVICE in ('cuda:0', 'cpu', 'cuda:1',)

# file resources
file_resources = FileResources(root=DATA_ROOT, crosswalk=CROSSWALK, baseline=not R2V,
                                model_name=GNN_MODEL, basename=DATASET)
print({
    'data_root': DATA_ROOT,
    'crosswalk': CROSSWALK,
    'baseline': not R2V,
    'model_name': GNN_MODEL,
    'device': SET_DEVICE,
    "dataset": DATASET
})

weighted_adj = file_resources.adj_path,
test_weighted_adj = file_resources.test_adj_path
BATCH_SIZE=256 * 3
NODE_TO_VEC_DIM=16
NUM_WORKERS=20
SET_DEVICE=SET_DEVICE
RV_NUM_WALKS=100
NODE_TO_VEC_EPOCHS=5

os.environ["SET_GPU"] = SET_DEVICE
import gc
from utils.network_splitter import NetworkTrainTestSplitterWithMST
import warnings
from utils import snakemake_utils

warnings.filterwarnings("ignore")
gc.enable()

window_length = 5
num_walks = 10
dim = 128
walk_length = 5

d = snakemake_utils.get_dataset(DATASET)
edge_index, num_nodes = d.edge_index, d.X.shape[0]
n = NetworkTrainTestSplitterWithMST(num_nodes=num_nodes, edge_list=edge_index)
n.train_test_split()
X = snakemake_utils.store_crosswalk_weights(
    file_path=weighted_adj,
    crosswalk=CROSSWALK,
    embedding_dim=NODE_TO_VEC_DIM,
    num_nodes=num_nodes,
    edge_index=n.train_edges,
    context_size=2,
    walk_length=walk_length,
    group_membership=d.get_grouped_col()
)
X = snakemake_utils.store_crosswalk_weights(
    file_path=test_weighted_adj,
    crosswalk=CROSSWALK,
    embedding_dim=NODE_TO_VEC_DIM,
    num_nodes=num_nodes,
    edge_index=n.test_edges,
    context_size=2,
    walk_length=walk_length,
    group_membership=d.get_grouped_col()
)


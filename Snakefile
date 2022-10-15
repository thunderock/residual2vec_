import os
from os.path import join as j
from utils.snakemake_utils import get_string_boolean, FileResources
import wandb
# os.environ["CUDA_VISIBLE_DEVICES"]=""
# config = {'gnn_model': 'gat', 'crosswalk': 'true',
#           'r2v': 'true', 'dataset': 'pokec', 'device': 'cuda:0'}


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
DATA_ROOT = config.get("root", "data")

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

residual2vec_training_epochs = {
  'pokec': 2,
  'small_pokec': 4,
  'airport': 100,
  'polblog': 200,
  'polbook': 200
}

num_gnn_layers = {
    'pokec': 5,
    'small_pokec': 5,
    'airport': 3,
    'polblog': 3,
    'polbook': 3
}
rule train_gnn:
    input:
        node2vec_weights=file_resources.node2vec_embs,
        weighted_adj=file_resources.adj_path
    output:
        model_weights = file_resources.model_weights
    threads: 20
    params:
        BATCH_SIZE = 256 * 3,
        NODE_TO_VEC_DIM= 16,
        NUM_WORKERS = 20,
        SET_DEVICE = SET_DEVICE,
        RV_NUM_WALKS= 100
    run:
        os.environ["SET_GPU"] = params.SET_DEVICE
        import torch
        from dataset import triplet_dataset
        from torch_geometric.utils import negative_sampling
        import gc
        from utils.link_prediction import GCNLinkPrediction, GATLinkPrediction
        import residual2vec as rv
        import warnings
        from utils import snakemake_utils
        warnings.filterwarnings("ignore")
        gc.enable()
        window_length = 5
        num_walks = 10
        dim = 128
        walk_length = 80
        # get edge index from training set
        edge_index = snakemake_utils.get_edge_index_from_sparse_path(input.weighted_adj)
        num_nodes = snakemake_utils.get_num_nodes_from_adj(input.weighted_adj)
        labels = snakemake_utils.get_dataset(DATASET).get_grouped_col()
        sampler = negative_sampling
        if R2V:
            sbm = triplet_dataset.SbmSamplerWrapper(
                adj_path=input.weighted_adj,
                group_membership=labels,
                window_length=1,
                padding_id=num_nodes,
                num_walks=params.RV_NUM_WALKS,
                use_weights=CROSSWALK,
                num_edges = edge_index.shape[1]
            )

            # dont use sbm negative sampler for training
            sampler = sbm.sample_neg_edges
            print("using de biased walk")
        X = snakemake_utils.get_node2vec_trained_get_embs(
            file_path=input.node2vec_weights
        )
        model = rv.residual2vec_sgd(
            noise_sampler=False,
            window_length=window_length,
            num_walks=num_walks,
            walk_length=walk_length,
            batch_size=params.BATCH_SIZE,
        ).fit()
        # X = torch.cat([X, d.X], dim=1)
        d = triplet_dataset.TripletGraphDataset(X=X, edge_index=edge_index, sampler=sampler)
        wandb.init(project=DATASET,name="DATA_ROOT={}_MODEL={}_CROSSWALK={}_R2V={}".format(DATA_ROOT, GNN_MODEL, CROSSWALK, R2V))
        dataloader = triplet_dataset.NeighborEdgeSampler(d, batch_size=model.batch_size, shuffle=True, num_workers=params.NUM_WORKERS, pin_memory=True)
        if GNN_MODEL == 'gat':
            m = GATLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=num_gnn_layers[DATASET],num_embeddings=X.shape[1])
        elif GNN_MODEL == 'gcn':
            m = GCNLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=num_gnn_layers[DATASET],num_embeddings=X.shape[1])
        else:
            raise ValueError("GNN_MODEL must be either gat or gcn")

        model.transform(model=m, dataloader=dataloader, epochs=residual2vec_training_epochs[DATASET])
        wandb.finish(exit_code=0)
        torch.save(m.state_dict(), output.model_weights)

rule generate_crosswalk_weights:
    output:
        weighted_adj =  FileResources(root=DATA_ROOT, crosswalk=True, baseline=not R2V,
                                model_name=GNN_MODEL, basename=DATASET).adj_path,
        test_weighted_adj = FileResources(root=DATA_ROOT, crosswalk=True, baseline=not R2V,
                                model_name=GNN_MODEL, basename=DATASET).test_adj_path,
        unweighted_adj =  FileResources(root=DATA_ROOT,crosswalk=False,baseline=not R2V,
        model_name=GNN_MODEL,basename=DATASET).adj_path,
        test_unweighted_adj = FileResources(root=DATA_ROOT,crosswalk=False,baseline=not R2V,
        model_name=GNN_MODEL,basename=DATASET).test_adj_path,

    params:
        BATCH_SIZE=256 * 3,
        NODE_TO_VEC_DIM=16,
        NUM_WORKERS=20,
        SET_DEVICE=SET_DEVICE,
        RV_NUM_WALKS=100
    threads: 20
    run:
        os.environ["SET_GPU"] = params.SET_DEVICE
        import gc
        from utils.network_splitter import NetworkTrainTestSplitterWithMST
        import warnings
        from utils import snakemake_utils

        warnings.filterwarnings("ignore")
        gc.enable()

        window_length = 5
        num_walks = 10
        dim = 128
        walk_length = 80
        # this is super hack here
        d = snakemake_utils.get_dataset(DATASET)
        edge_index, num_nodes = d.edge_index, d.X.shape[0]
        n = NetworkTrainTestSplitterWithMST(num_nodes=num_nodes, edge_list=edge_index)
        # nodes are not made symmetric here
        n.train_test_split()
        # nodes are made symmetric here

        snakemake_utils.store_crosswalk_weights(
            file_path=output.weighted_adj,
            crosswalk=True,
            embedding_dim=params.NODE_TO_VEC_DIM,
            num_nodes=num_nodes,
            edge_index=n.train_edges,
            group_membership=d.get_grouped_col()
        )
        snakemake_utils.store_crosswalk_weights(
            file_path=output.test_weighted_adj,
            crosswalk=True,
            embedding_dim=params.NODE_TO_VEC_DIM,
            num_nodes=num_nodes,
            edge_index=n.train_edges,
            group_membership=d.get_grouped_col()
        )
        snakemake_utils.store_crosswalk_weights(
            file_path=output.unweighted_adj,
            crosswalk=False,
            embedding_dim=params.NODE_TO_VEC_DIM,
            num_nodes=num_nodes,
            edge_index=n.train_edges,
            group_membership=d.get_grouped_col()
        )
        snakemake_utils.store_crosswalk_weights(
            file_path=output.test_unweighted_adj,
            crosswalk=False,
            embedding_dim=params.NODE_TO_VEC_DIM,
            num_nodes=num_nodes,
            edge_index=n.test_edges,
            group_membership=d.get_grouped_col()
        )


rule train_node_2_vec:
    # snakemake -R --until train_gnn_with_nodevec_unweighted_baseline  -call --config env=local model=gat
    input:
        weighted_adj=file_resources.adj_path
    output:
        node2vec_weights = file_resources.node2vec_embs
    threads: 20
    params:
        BATCH_SIZE = 256 * 3,
        NODE_TO_VEC_DIM= 16,
        NUM_WORKERS = 20,
        SET_DEVICE = SET_DEVICE,
        RV_NUM_WALKS= 100
    run:
        os.environ["SET_GPU"] = params.SET_DEVICE
        from dataset import triplet_dataset
        import gc
        import warnings
        from utils import snakemake_utils
        warnings.filterwarnings("ignore")
        gc.enable()

        edge_index = snakemake_utils.get_edge_index_from_sparse_path(input.weighted_adj)
        num_nodes = snakemake_utils.get_num_nodes_from_adj(input.weighted_adj)

        labels = snakemake_utils.get_dataset(DATASET).get_grouped_col()
        snakemake_utils.train_node2vec_get_embs(
            edge_index=edge_index,
            file_path=output.node2vec_weights,
            crosswalk=CROSSWALK,
            embedding_dim=params.NODE_TO_VEC_DIM,
            num_nodes=num_nodes,
            weighted_adj_path=input.weighted_adj,
            group_membership=labels
        )


rule generate_node_embeddings:
    input:
        node2vec_weights = file_resources.node2vec_embs,
        model_weights = file_resources.model_weights,
        weighted_adj = file_resources.test_adj_path
    output:
        embs_file = file_resources.embs_file
    params:
        BATCH_SIZE = 256 * 3,
        NODE_TO_VEC_DIM= 16,
        NUM_WORKERS = 20,
        SET_DEVICE = SET_DEVICE,
        RV_NUM_WALKS= 100
    threads: 20
    run:
        os.environ["SET_GPU"] = params.SET_DEVICE
        import torch
        import numpy as np
        from dataset import triplet_dataset, pokec_data
        from utils.config import DEVICE
        import gc
        from utils.link_prediction import GCNLinkPrediction, GATLinkPrediction
        from torch_geometric.utils import negative_sampling
        import residual2vec as rv
        import warnings
        from utils import snakemake_utils
        from tqdm import tqdm

        warnings.filterwarnings("ignore")
        gc.enable()
        window_length = 5
        num_walks = 10
        dim = 128
        walk_length = 80
        edge_index = snakemake_utils.get_edge_index_from_sparse_path(input.weighted_adj)
        num_nodes = snakemake_utils.get_num_nodes_from_adj(input.weighted_adj)
        sampler = negative_sampling
        labels = snakemake_utils.get_dataset(DATASET).get_grouped_col()
        if R2V:
            sbm = triplet_dataset.SbmSamplerWrapper(
                adj_path=input.weighted_adj,
                group_membership=labels,
                window_length=1,
                padding_id=num_nodes,
                num_walks=params.RV_NUM_WALKS,
                use_weights=CROSSWALK,
                num_edges = edge_index.shape[1]
            )

            # dont use sbm negative sampler for training
            sampler = sbm.sample_neg_edges
            print("using de biased walk")

        X = snakemake_utils.get_node2vec_trained_get_embs(
            file_path=input.node2vec_weights,
        )
        model = rv.residual2vec_sgd(
            noise_sampler=False,
            window_length=window_length,
            num_walks=num_walks,
            walk_length=walk_length,
            batch_size=params.BATCH_SIZE,
        ).fit()
        # X = torch.cat([X, d.X], dim=1)
        d = triplet_dataset.TripletGraphDataset(X=X,edge_index=edge_index, sampler=sampler)
        dataloader = triplet_dataset.NeighborEdgeSampler(d,batch_size=model.batch_size,shuffle=False,num_workers=params.NUM_WORKERS,pin_memory=True,transforming=True)
        if GNN_MODEL == 'gat':
            m = GATLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=num_gnn_layers[DATASET],num_embeddings=
            X.shape[1])
        elif GNN_MODEL == 'gcn':
            m = GCNLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=num_gnn_layers[DATASET],num_embeddings=
            X.shape[1])
        else:
            raise ValueError("GNN_MODEL must be either gat or gcn")
        m = m.to(DEVICE)
        m.load_state_dict(torch.load(input.model_weights))
        embs = torch.zeros((num_nodes, 128))
        batch_size = model.batch_size
        m.eval()
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataloader,desc="Generating node embeddings")):
                a, p, n = batch
                a = m.forward_i(a)# , m.forward_o(p), m.forward_o(n)
                a = a.detach().cpu()# , p.detach().cpu(), n.detach().cpu()
                embs[idx * batch_size:(idx + 1) * batch_size, :] = a # torch.cat((a, p, n),dim=1)
        np.save(output.embs_file,embs.numpy())


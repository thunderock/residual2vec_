import os
from os.path import join as j
from utils.snakemake_utils import get_string_boolean, FileResources
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

rule train_gnn:
    # snakemake -R --until train_gnn_with_nodevec_unweighted_baseline  -call --config env=local model=gat
    input:
        node2vec_weights=file_resources.node2vec_weights,
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
        from models.weighted_node2vec import UnWeightedNode2Vec
        from dataset import triplet_dataset, pokec_data
        from utils.config import DEVICE
        import gc
        from utils.link_prediction import GCNLinkPrediction, GATLinkPrediction
        import residual2vec as rv
        import warnings
        from torch_geometric.utils import negative_sampling
        from utils import snakemake_utils
        warnings.filterwarnings("ignore")
        gc.enable()
        window_length = 5
        num_walks = 10
        dim = 128
        walk_length = 5

        d = snakemake_utils.get_dataset(DATASET)
        edge_index, num_nodes = d.edge_index, d.X.shape[0]
        sampler = negative_sampling
        if R2V:
            sbm = triplet_dataset.SbmSamplerWrapper(
                adj_path=input.weighted_adj,
                group_membership=d.get_grouped_col(),
                window_length=1,
                padding_idx=num_nodes,
                num_walks=params.RV_NUM_WALKS,
                num_nodes=edge_index.shape[1],
                use_weights=CROSSWALK
            )
            edge_index = sbm.edge_index
            sampler = sbm.sample_neg_edges
            print("using de biased walk")
        X = snakemake_utils.get_node2vec_trained_get_embs(
            file_path=input.node2vec_weights,
            crosswalk=CROSSWALK,
            embedding_dim=params.NODE_TO_VEC_DIM,
            num_nodes=num_nodes,
            context_size=2,
            walk_length=walk_length,
            weighted_adj_path=input.weighted_adj,
            group_membership=d.get_grouped_col()
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
        dataloader = triplet_dataset.NeighborEdgeSampler(d, batch_size=model.batch_size, shuffle=True, num_workers=params.NUM_WORKERS, pin_memory=True)
        if GNN_MODEL == 'gat':
            m = GATLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=X.shape[1])
        elif GNN_MODEL == 'gcn':
            m = GCNLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=X.shape[1])
        else:
            raise ValueError("GNN_MODEL must be either gat or gcn")

        model.transform(model=m, dataloader=dataloader)
        torch.save(m.state_dict(), output.model_weights)

rule generate_crosswalk_weights:
    output:
        weighted_adj = file_resources.adj_path
    params:
        BATCH_SIZE=256 * 3,
        NODE_TO_VEC_DIM=16,
        NUM_WORKERS=20,
        SET_DEVICE=SET_DEVICE,
        RV_NUM_WALKS=100,
        NODE_TO_VEC_EPOCHS=5,
    threads: 20
    run:
        os.environ["SET_GPU"] = params.SET_DEVICE
        import torch
        from dataset import triplet_dataset, pokec_data
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
        walk_length = 5

        d = snakemake_utils.get_dataset(DATASET)
        edge_index, num_nodes = d.edge_index, d.X.shape[0]
        if R2V:
            sbm = triplet_dataset.SbmSamplerWrapper(
                adj_path=output.weighted_adj,
                group_membership=d.get_grouped_col(),
                window_length=1,
                padding_idx=num_nodes,
                num_walks=params.RV_NUM_WALKS,
                num_nodes=edge_index.shape[1],
                use_weights=CROSSWALK
            )
            edge_index = sbm.edge_index
            print("using de biased walk")
        X = snakemake_utils.store_crosswalk_weights(
            file_path=output.weighted_adj,
            crosswalk=CROSSWALK,
            embedding_dim=params.NODE_TO_VEC_DIM,
            num_nodes=num_nodes,
            edge_index=edge_index,
            context_size=2,
            walk_length=walk_length,
            group_membership=d.get_grouped_col()
        )


rule train_node_2_vec:
    # snakemake -R --until train_gnn_with_nodevec_unweighted_baseline  -call --config env=local model=gat
    input:
        weighted_adj=file_resources.adj_path
    output:
        node2vec_weights = file_resources.node2vec_weights,
    threads: 20
    params:
        BATCH_SIZE = 256 * 3,
        NODE_TO_VEC_DIM= 16,
        NUM_WORKERS = 20,
        SET_DEVICE = SET_DEVICE,
        RV_NUM_WALKS= 100,
        NODE_TO_VEC_EPOCHS= 5,
    run:
        os.environ["SET_GPU"] = params.SET_DEVICE
        import torch
        from dataset import triplet_dataset, pokec_data
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
        walk_length = 5

        d = snakemake_utils.get_dataset(DATASET)
        edge_index, num_nodes = d.edge_index, d.X.shape[0]
        if R2V:
            sbm = triplet_dataset.SbmSamplerWrapper(
                adj_path=input.weighted_adj,
                group_membership=d.get_grouped_col(),
                window_length=1,
                padding_idx=num_nodes,
                num_walks=params.RV_NUM_WALKS,
                num_nodes=edge_index.shape[1],
                use_weights=CROSSWALK
            )
            edge_index = sbm.edge_index
            print("using de biased walk")
        snakemake_utils.train_node2vec_get_embs(
            file_path=output.node2vec_weights,
            batch_size=params.BATCH_SIZE,
            num_workers=params.NUM_WORKERS,
            epochs=params.NODE_TO_VEC_EPOCHS,
            crosswalk=CROSSWALK,
            embedding_dim=params.NODE_TO_VEC_DIM,
            num_nodes=num_nodes,
            context_size=2,
            edge_index=edge_index,
            walk_length=walk_length,
            weighted_adj_path=input.weighted_adj if CROSSWALK else None,
            group_membership=d.get_grouped_col()
        )


rule generate_node_embeddings:
    input:
        node2vec_weights = file_resources.node2vec_weights,
        model_weights = file_resources.model_weights,
        weighted_adj = file_resources.adj_path
    output:
        embs_file = file_resources.embs_file
    params:
        BATCH_SIZE = 256 * 3,
        NODE_TO_VEC_DIM= 16,
        NUM_WORKERS = 20,
        SET_DEVICE = SET_DEVICE,
        RV_NUM_WALKS= 100,
        NODE_TO_VEC_EPOCHS= 5
    threads: 20
    run:
        os.environ["SET_GPU"] = params.SET_DEVICE
        import torch
        import numpy as np
        from dataset import triplet_dataset, pokec_data
        from utils.config import DEVICE
        import gc
        from utils.link_prediction import GCNLinkPrediction, GATLinkPrediction
        import residual2vec as rv
        import warnings
        from torch_geometric.utils import negative_sampling
        from utils import snakemake_utils
        from tqdm import tqdm

        warnings.filterwarnings("ignore")
        gc.enable()
        window_length = 5
        num_walks = 10
        dim = 128
        walk_length = 5

        d = snakemake_utils.get_dataset(DATASET)
        edge_index, num_nodes = d.edge_index, d.X.shape[0]
        sampler = negative_sampling
        if R2V:
            sbm = triplet_dataset.SbmSamplerWrapper(
                adj_path=input.weighted_adj,
                group_membership=d.get_grouped_col(),
                window_length=1,
                padding_idx=num_nodes,
                num_walks=params.RV_NUM_WALKS,
                num_nodes=edge_index.shape[1],
                use_weights=CROSSWALK
            )
            edge_index = sbm.edge_index
            sampler = sbm.sample_neg_edges
            print("using de biased walk")
        X = snakemake_utils.get_node2vec_trained_get_embs(
            file_path=input.node2vec_weights,
            crosswalk=CROSSWALK,
            embedding_dim=params.NODE_TO_VEC_DIM,
            num_nodes=num_nodes,
            context_size=2,
            walk_length=walk_length,
            weighted_adj_path=input.weighted_adj,
            group_membership=d.get_grouped_col()
        )
        model = rv.residual2vec_sgd(
            noise_sampler=False,
            window_length=window_length,
            num_walks=num_walks,
            walk_length=walk_length,
            batch_size=params.BATCH_SIZE,
        ).fit()
        # X = torch.cat([X, d.X], dim=1)
        d = triplet_dataset.TripletGraphDataset(X=X,edge_index=edge_index,sampler=sampler)
        dataloader = triplet_dataset.NeighborEdgeSampler(d,batch_size=model.batch_size,shuffle=True,num_workers=params.NUM_WORKERS,pin_memory=True)
        if GNN_MODEL == 'gat':
            m = GATLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=
            X.shape[1])
        elif GNN_MODEL == 'gcn':
            m = GCNLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=
            X.shape[1])
        else:
            raise ValueError("GNN_MODEL must be either gat or gcn")
        m = m.to(DEVICE)
        embs = torch.zeros((num_nodes, 128 * 3))
        batch_size = model.batch_size
        m.eval()
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataloader,desc="Generating node embeddings")):
                a, p, n = batch
                a, p, n = m.forward_i(a), m.forward_o(p), m.forward_o(n)
                a, p, n = a.detach().cpu(), p.detach().cpu(), n.detach().cpu()
                embs[idx * batch_size:(idx + 1) * batch_size, :] = torch.cat((a, p, n),dim=1)
        np.save(output.embs_file,embs.numpy())
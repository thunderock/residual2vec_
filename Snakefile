import os
from utils.snakemake_utils import get_string_boolean, FileResources
import wandb
from utils.config import R2V_TRAINING_EPOCHS, NUM_NEGATIVE_SAMPLING
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

FAIRWALK = config.get('fairwalk', 'false')
FAIRWALK = get_string_boolean(FAIRWALK)

NODE2VEC = config.get('node2vec', 'false')
NODE2VEC = get_string_boolean(NODE2VEC)

R2V = config.get('r2v', 'false')
R2V = get_string_boolean(R2V)

SET_DEVICE = config.get('device', 'cuda:0')
# DATA_ROOT = "/data/sg/ashutiwa/residual2vec_"
# if ENV in ('local', 'carbonate'):
DATA_ROOT = config.get("root", "data")

# variables sanity check
assert GNN_MODEL in ('gat', 'gcn', 'word2vec')
assert DATASET in ('pokec', 'small_pokec', 'airport', 'polblog', 'polbook')
assert CROSSWALK in (True, False)
assert R2V in (True, False)
assert SET_DEVICE in ('cuda:0', 'cpu', 'cuda:1',)
assert FAIRWALK in (True, False)
assert not (CROSSWALK and FAIRWALK), "CROSSWALK and FAIRWALK cannot be both True"
assert NODE2VEC in (True, False)

# file resources
file_resources = FileResources(root=DATA_ROOT, crosswalk=CROSSWALK, fairwalk=FAIRWALK,
    node2vec=NODE2VEC, r2v=R2V, dataset=DATASET, model_name=GNN_MODEL)
print(config)


rule train_gnn:
    input:
        feature_weights=file_resources.feature_embs,
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
        import residual2vec as rv
        import numpy as np
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
        X = snakemake_utils.get_feature_trained_get_embs(
            file_path=input.feature_weights
        )
        model = rv.residual2vec_sgd(
            noise_sampler=False,
            window_length=window_length,
            num_walks=num_walks,
            walk_length=walk_length,
            batch_size=params.BATCH_SIZE,
        ).fit()
        d = triplet_dataset.TripletGraphDataset(X=X, edge_index=edge_index, sampler=sampler, num_neg_sampling=NUM_NEGATIVE_SAMPLING[DATASET])
        wandb.init(project=DATASET,name="DATA_ROOT={}_MODEL={}_CROSSWALK={}_FAIRWALK={}_NODE2VEC={}_R2V={}".format(DATA_ROOT, GNN_MODEL, CROSSWALK, FAIRWALK, NODE2VEC, R2V))
        dataloader = triplet_dataset.NeighborEdgeSampler(d, batch_size=model.batch_size, shuffle=True, num_workers=params.NUM_WORKERS, pin_memory=True)
        if GNN_MODEL in ['gat', 'gcn']:
            m = snakemake_utils.get_gnn_model(
                model_name=GNN_MODEL,
                num_features=X.shape[1],
                emb_dim=dim,
                dataset=DATASET,
                num_layers=None,
                learn_outvec=False
            )
        elif GNN_MODEL == 'word2vec':
            # create adj matrix
            adj_mat = snakemake_utils.get_adj_mat_from_path(input.weighted_adj)
            # first create noise sampler
            from residual2vec.node_samplers import SBMNodeSampler
            from torch.utils.data import DataLoader
            from residual2vec.word2vec import Word2Vec
            if R2V:
                # pick sbm node sampler
                noise_sampler = SBMNodeSampler(
                    window_length=1,
                    group_membership=labels,
                    dcsbm=True
                )
            else:
                noise_sampler = SBMNodeSampler(
                    window_length=1,
                    dcsbm=True
                )
            model = rv.residual2vec_sgd(
                noise_sampler=noise_sampler,
                window_length=1,
                num_walks=num_walks,
                walk_length=walk_length,
                batch_size=params.BATCH_SIZE,
            ).fit(adjmat=adj_mat)
            adjusted_num_walks = np.ceil(
                num_walks
                * np.maximum(
                    1,
                    model.batch_size
                    * model.miniters
                    / (model.n_nodes * num_walks * walk_length),
                )
            ).astype(int)
            d = rv.TripletSimpleDataset(
                adjmat=model.adjmat,
                num_walks=adjusted_num_walks,
                window_length=model.window_length,
                noise_sampler=model.sampler,
                padding_id=model.n_nodes,
                walk_length=model.walk_length,
                p=model.p,
                q=model.q,
                buffer_size=model.buffer_size,
                context_window_type=model.context_window_type,
            )
            dataloader = DataLoader(
                d,
                batch_size=model.batch_size,
                shuffle=True,
                num_workers=params.NUM_WORKERS,
                pin_memory=True,
            )
            m = Word2Vec(
                vocab_size=num_nodes + 1,
                embedding_size=128,
                padding_idx=num_nodes,
                learn_outvec=False
            )
        else:
            raise ValueError("GNN_MODEL must be either gat or gcn")

        model.transform(model=m, dataloader=dataloader, epochs=R2V_TRAINING_EPOCHS[DATASET])
        wandb.finish(exit_code=0)
        torch.save(m.state_dict(), output.model_weights)

rule generate_crosswalk_weights:
    output:
        # crosswalk_weighted_adj =  FileResources(root=DATA_ROOT, fairwalk=False, crosswalk=True, r2v="doesnt_matter", model_name="doesnt_matter",
        #     node2vec="doesnt_matter", dataset=DATASET).adj_path,
        fairwalk_weighted_adj =  FileResources(root=DATA_ROOT,fairwalk=True,crosswalk=False,r2v="doesnt_matter",model_name="doesnt_matter",
            node2vec="doesnt_matter", dataset=DATASET).adj_path,
        unweighted_adj =  FileResources(root=DATA_ROOT,fairwalk=False,crosswalk=False,r2v="doesnt_matter",model_name="doesnt_matter",
            node2vec="doesnt_matter",dataset=DATASET).adj_path,
        test_adj =  FileResources(root=DATA_ROOT, fairwalk="doesnt_matter", crosswalk="doesnt_matter", r2v="doesnt_matter", model_name="doesnt_matter",
            node2vec="doesnt_matter", dataset=DATASET).test_adj_path

    params:
        BATCH_SIZE=256 * 3,
        NODE_TO_VEC_DIM=16,
        NUM_WORKERS=20,
        SET_DEVICE=SET_DEVICE,
        RV_NUM_WALKS=100
    threads: 10
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

        # snakemake_utils.store_weighted_adj(
        #     file_path=output.crosswalk_weighted_adj,
        #     crosswalk=True,
        #     fairwalk=False,
        #     embedding_dim=params.NODE_TO_VEC_DIM,
        #     num_nodes=num_nodes,
        #     edge_index=n.train_edges,
        #     group_membership=d.get_grouped_col()
        # )
        snakemake_utils.store_weighted_adj(
            file_path=output.fairwalk_weighted_adj,
            crosswalk=False,
            fairwalk=True,
            embedding_dim=params.NODE_TO_VEC_DIM,
            num_nodes=num_nodes,
            edge_index=n.train_edges,
            group_membership=d.get_grouped_col()
        )
        snakemake_utils.store_weighted_adj(
            file_path=output.unweighted_adj,
            crosswalk=False,
            fairwalk=False,
            embedding_dim=params.NODE_TO_VEC_DIM,
            num_nodes=num_nodes,
            edge_index=n.train_edges,
            group_membership=d.get_grouped_col()
        )
        snakemake_utils.store_weighted_adj(
            file_path=output.test_adj,
            crosswalk=False,
            fairwalk=False,
            embedding_dim=params.NODE_TO_VEC_DIM,
            num_nodes=num_nodes,
            edge_index=n.train_edges,
            group_membership=d.get_grouped_col()
        )

rule train_features_2_vec:
    # snakemake -R --until train_gnn_with_nodevec_unweighted_baseline  -call --config env=local model=gat
    input:
        weighted_adj=file_resources.adj_path
    output:
        features = file_resources.feature_embs
    threads: 20
    params:
        BATCH_SIZE = 256 * 3,
        NODE_TO_VEC_DIM= 16,
        NUM_WORKERS = 20,
        SET_DEVICE = SET_DEVICE,
        RV_NUM_WALKS= 100
    run:
        os.environ["SET_GPU"] = params.SET_DEVICE
        import gc
        import warnings
        from utils import snakemake_utils
        warnings.filterwarnings("ignore")
        gc.enable()

        edge_index = snakemake_utils.get_edge_index_from_sparse_path(input.weighted_adj)
        num_nodes = snakemake_utils.get_num_nodes_from_adj(input.weighted_adj)

        labels = snakemake_utils.get_dataset(DATASET).get_grouped_col()
        if NODE2VEC:
            snakemake_utils.train_node2vec_get_embs(
                edge_index=edge_index,
                file_path=output.features,
                crosswalk=CROSSWALK,
                fairwalk=FAIRWALK,
                embedding_dim=params.NODE_TO_VEC_DIM,
                num_nodes=num_nodes,
                weighted_adj_path=input.weighted_adj,
                group_membership=labels
            )
        else:
            snakemake_utils.train_deepwalk_get_embs(
                edge_index=edge_index,
                file_path=output.features,
                crosswalk=CROSSWALK,
                fairwalk=FAIRWALK,
                embedding_dim=params.NODE_TO_VEC_DIM,
                num_nodes=num_nodes,
                weighted_adj_path=input.weighted_adj,
                group_membership=labels
            )



rule generate_node_embeddings:
    input:
        features = file_resources.feature_embs,
        model_weights = file_resources.model_weights,
        weighted_adj = file_resources.adj_path
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
        from dataset import triplet_dataset
        from utils.config import DEVICE
        import gc
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

        X = snakemake_utils.get_feature_trained_get_embs(
            file_path=input.features,
        )
        model = rv.residual2vec_sgd(
            noise_sampler=False,
            window_length=window_length,
            num_walks=num_walks,
            walk_length=walk_length,
            batch_size=params.BATCH_SIZE,
        ).fit()
        # X = torch.cat([X, d.X], dim=1)
        d = triplet_dataset.TripletGraphDataset(X=X,edge_index=edge_index, sampler=sampler, num_neg_sampling=NUM_NEGATIVE_SAMPLING[DATASET])
        dataloader = triplet_dataset.NeighborEdgeSampler(d,batch_size=model.batch_size,shuffle=False,num_workers=params.NUM_WORKERS,pin_memory=True,transforming=True)
        if GNN_MODEL in ['gat', 'gcn']:
            m = snakemake_utils.get_gnn_model(
                model_name=GNN_MODEL,
                num_features=X.shape[1],
                emb_dim=dim,
                dataset=DATASET,
                num_layers=None,
                learn_outvec=False
            )

        elif GNN_MODEL == 'word2vec':
            # create adj matrix
            adj_mat = snakemake_utils.get_adj_mat_from_path(input.weighted_adj)
            # first create noise sampler
            from residual2vec.node_samplers import SBMNodeSampler
            from torch.utils.data import DataLoader
            from residual2vec.word2vec import Word2Vec
            if R2V:
                # pick sbm node sampler
                noise_sampler = SBMNodeSampler(
                    window_length=1,
                    group_membership=labels,
                    dcsbm=True
                )
            else:
                noise_sampler = SBMNodeSampler(
                    window_length=1,
                    dcsbm=True
                )
            model = rv.residual2vec_sgd(
                noise_sampler=noise_sampler,
                window_length=1,
                num_walks=num_walks,
                walk_length=walk_length,
                batch_size=params.BATCH_SIZE,
            ).fit(adjmat=adj_mat)
            adjusted_num_walks = np.ceil(
                num_walks
                * np.maximum(
                    1,
                    model.batch_size
                    * model.miniters
                    / (model.n_nodes * num_walks * walk_length),
                )
            ).astype(int)
            d = rv.TripletSimpleDataset(
                adjmat=model.adjmat,
                num_walks=adjusted_num_walks,
                window_length=model.window_length,
                noise_sampler=model.sampler,
                padding_id=model.n_nodes,
                walk_length=model.walk_length,
                p=model.p,
                q=model.q,
                buffer_size=model.buffer_size,
                context_window_type=model.context_window_type,
            )
            dataloader = DataLoader(
                d,
                batch_size=model.batch_size,
                shuffle=False,
                num_workers=params.NUM_WORKERS,
                pin_memory=True,
            )
            m = Word2Vec(
                vocab_size=num_nodes + 1,
                embedding_size=128,
                padding_idx=num_nodes,
                learn_outvec=False,
            )
        else:
            raise ValueError("GNN_MODEL must be either gat or gcn")
        m = m.to(DEVICE)
        m.load_state_dict(torch.load(input.model_weights))

        embs = torch.zeros((num_nodes, 128))
        if GNN_MODEL == 'word2vec':
            embs = torch.from_numpy(m.ivectors.weight.data.cpu().numpy()[:num_nodes, :])
        batch_size = model.batch_size
        m.eval()
        if GNN_MODEL != 'word2vec':
            with torch.no_grad():
                for idx, batch in enumerate(tqdm(dataloader,desc="Generating node embeddings")):
                    a, _, _ = batch
                    a = m.forward_i(a).detach().cpu()
                    nodes_remaining = num_nodes - (idx * batch_size)
                    if nodes_remaining < batch_size:
                        embs[idx * batch_size:, :] = a[-nodes_remaining:, :]
                        break
                    else:
                        embs[idx * batch_size:(idx + 1) * batch_size, :] = a
        np.save(output.embs_file,embs.numpy())

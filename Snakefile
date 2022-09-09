import os
from os.path import join as j
# os.environ["CUDA_VISIBLE_DEVICES"]=""
# config = {"env": 'local', 'model': 'gat'}
ENV = config.get('env', 'remote')
DATA_ROOT = "/data/sg/ashutiwa/residual2vec_"
if ENV in ('local', 'carbonate'):
    DATA_ROOT = "data"

GNN_MODEL = config.get('model', "gat")
# CROSSWALK = config.get('crosswalk', "false")

rule train_gnn_with_nodevec_unweighted_baseline:
    # snakemake -R --until train_gnn_with_nodevec_unweighted_baseline  -call --config env=local model=gat
    output:
        model_weights = j(DATA_ROOT, "pokec_{}_nodevec.h5".format(GNN_MODEL)),
        node2vec_weights = j(DATA_ROOT, "pokec_{}_node2vec.h5".format(GNN_MODEL))
    threads: 4 if ENV == 'local' else 10
    params:
        BATCH_SIZE = 128,
        NODE_TO_VEC_DIM = 16,
        NODE_TO_VEC_EPOCHS = 5,
        NUM_WORKERS = 4 if ENV == 'local' else 10,
        SET_DEVICE = "cuda:0" if GNN_MODEL == "gat" else "cuda:1"
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
        warnings.filterwarnings("ignore")
        gc.enable()
        window_length = 5
        num_walks = 10
        dim = 128
        walk_length = 5

        d = pokec_data.PokecDataFrame()
        edge_index, num_nodes = d.edge_index, d.X.shape[0]
        node_to_vec = UnWeightedNode2Vec(
            num_nodes=num_nodes,
            edge_index=edge_index,
            embedding_dim=params.NODE_TO_VEC_DIM,
            walk_length=walk_length,
            context_size=2,).to(DEVICE)
        model = rv.residual2vec_sgd(
            noise_sampler=False,
            window_length=window_length,
            num_walks=num_walks,
            walk_length=walk_length
        ).fit()
        loader = node_to_vec.loader(batch_size=params.BATCH_SIZE,
            shuffle=True,
            num_workers=params.NUM_WORKERS,)
        optimizer = torch.optim.Adam(list(node_to_vec.parameters()),lr=0.01)
        X = node_to_vec.train_and_get_embs(loader, optimizer, params.NODE_TO_VEC_EPOCHS, str(output.node2vec_weights))
        X = torch.cat([X, d.X], dim=1)
        d = triplet_dataset.TripletGraphDataset(X=X, edge_index=edge_index,)
        dataloader = triplet_dataset.NeighborEdgeSampler(d, batch_size=model.batch_size, shuffle=True, num_workers=params.NUM_WORKERS, pin_memory=True)
        if GNN_MODEL == 'gat':
            m = GATLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=params.NODE_TO_VEC_DIM + 5)
        elif GNN_MODEL == 'gcn':
            m = GCNLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=params.NODE_TO_VEC_DIM + 5)
        else:
            raise ValueError("GNN_MODEL must be either gat or gcn")

        model.transform(model=m, dataloader=dataloader)
        torch.save(m.state_dict(), str(output.model_weights))

rule train_gnn_with_nodevec_crosswalk_baseline:
    # snakemake -R --until train_gnn_with_nodevec_unweighted_baseline  -call --config env=local model=gat
    output:
        model_weights = j(DATA_ROOT, "pokec_crosswalk_{}_nodevec.h5".format(GNN_MODEL)),
        node2vec_weights = j(DATA_ROOT, "pokec_crosswalk_{}_node2vec.h5".format(GNN_MODEL))
    input:
        weighted_adj = j(DATA_ROOT,"pokec_crosswalk_adj.npz"),
    threads: 4 if ENV == 'local' else 10
    params:
        BATCH_SIZE = 128,
        NODE_TO_VEC_DIM = 16,
        NODE_TO_VEC_EPOCHS = 5,
        NUM_WORKERS = 4 if ENV == 'local' else 10,
        SET_DEVICE = "cuda:0" if GNN_MODEL == "gat" else "cuda:1"
    run:
        os.environ["SET_GPU"] = params.SET_DEVICE
        import torch
        from models.weighted_node2vec import WeightedNode2Vec
        from dataset import triplet_dataset, pokec_data
        from utils.config import DEVICE
        import gc
        from utils.link_prediction import GCNLinkPrediction, GATLinkPrediction
        import residual2vec as rv
        import warnings
        warnings.filterwarnings("ignore")
        gc.enable()
        window_length = 5
        num_walks = 10
        dim = 128
        walk_length = 5

        d = pokec_data.PokecDataFrame()
        edge_index, num_nodes = d.edge_index, d.X.shape[0]
        node_to_vec = WeightedNode2Vec(
            num_nodes=num_nodes,
            group_membership=d.get_grouped_col(),
            weighted_adj=str(input.weighted_adj),
            edge_index=edge_index,
            embedding_dim=params.NODE_TO_VEC_DIM,
            walk_length=walk_length,
            context_size=2,).to(DEVICE)
        model = rv.residual2vec_sgd(
            noise_sampler=False,
            window_length=window_length,
            num_walks=num_walks,
            walk_length=walk_length
        ).fit()
        loader = node_to_vec.loader(batch_size=params.BATCH_SIZE,
            shuffle=True,
            num_workers=params.NUM_WORKERS,)
        optimizer = torch.optim.Adam(list(node_to_vec.parameters()),lr=0.01)
        X = node_to_vec.train_and_get_embs(loader, optimizer, params.NODE_TO_VEC_EPOCHS, str(output.node2vec_weights))
        X = torch.cat([X, d.X], dim=1)
        d = triplet_dataset.TripletGraphDataset(X=X, edge_index=edge_index,)
        dataloader = triplet_dataset.NeighborEdgeSampler(d, batch_size=model.batch_size, shuffle=True, num_workers=params.NUM_WORKERS, pin_memory=True)
        if GNN_MODEL == 'gat':
            m = GATLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=params.NODE_TO_VEC_DIM + 5)
        elif GNN_MODEL == 'gcn':
            m = GCNLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=params.NODE_TO_VEC_DIM + 5)
        else:
            raise ValueError("GNN_MODEL must be either gat or gcn")

        model.transform(model=m, dataloader=dataloader)
        torch.save(m.state_dict(), str(output.model_weights))

rule generate_embs:
    # snakemake -R --until generate_embs_crosswalk  -call --config env=local model=gat --nolock
    input:
        model_weights=j(DATA_ROOT,"pokec_{}.h5".format(GNN_MODEL)),
    output:
        embs_file = j(DATA_ROOT, "pokec_{}_embs_fixed.npy".format(GNN_MODEL))
    threads: 4 if ENV == 'local' else 20
    params:
        BATCH_SIZE= 128,
        NUM_WORKERS=4 if ENV == 'local' else 16,
        SET_DEVICE="cuda:0" if GNN_MODEL == 'gat' else "cuda:1"
    run:
        os.environ["SET_GPU"] = params.SET_DEVICE
        import numpy as np
        import torch
        from dataset import triplet_dataset, pokec_data
        from utils.config import DEVICE
        from utils.link_prediction import GATLinkPrediction, GCNLinkPrediction
        import residual2vec as rv
        import warnings
        import gc
        from tqdm import tqdm, trange

        warnings.filterwarnings("ignore")
        gc.enable()

        window_length = 5
        num_walks = 10
        dim = 128
        walk_length = 5

        d = pokec_data.PokecDataFrame()
        edge_index, num_nodes = d.edge_index, d.X.shape[0]

        model = rv.residual2vec_sgd(
            noise_sampler=rv.ConfigModelNodeSampler(),
            window_length=window_length,
            num_walks=num_walks,
            walk_length=walk_length
        ).fit()
        X = d.X
        d = triplet_dataset.TripletGraphDataset(X=X, edge_index=edge_index, )
        dataloader = triplet_dataset.NeighborEdgeSampler(d, batch_size=params.BATCH_SIZE, shuffle=False,
            num_workers=params.NUM_WORKERS, pin_memory=True, transforming=False)
        if GNN_MODEL == 'gat':
            m = GATLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=d.num_features)
        elif GNN_MODEL == 'gcn':
            m = GCNLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=d.num_features)
        else:
            raise ValueError("GNN_MODEL must be either gat or gcn")
        m.load_state_dict(torch.load(str(input.model_weights)))
        m = m.to(DEVICE)
        m.eval()

        embs = torch.zeros((num_nodes, 128 * 3))
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataloader,desc="Generating node embeddings")):
                a, p, n = batch
                a, p, n = m.forward_i(a), m.forward_o(p), m.forward_o(n)
                a, p, n = a.detach().cpu(), p.detach().cpu(), n.detach().cpu()
                embs[idx * params.BATCH_SIZE:(idx + 1) * params.BATCH_SIZE, :] = torch.cat((a, p, n),dim=1)
        np.save(str(output.embs_file),embs.numpy())

rule generate_embs_crosswalk:
    # snakemake -R --until generate_embs_crosswalk  -call --config env=local model=gat --nolock
    input:
        model_weights=j(DATA_ROOT,"pokec_crosswalk_{}_nodevec.h5".format(GNN_MODEL)),
        node2vec_weights=j(DATA_ROOT,"pokec_crosswalk_{}_node2vec.h5".format(GNN_MODEL)),
        weighted_adj= j(DATA_ROOT,"pokec_crosswalk_adj.npz"),
    output:
        embs_file = j(DATA_ROOT, "pokec_crosswalk_{}_embs_fixed.npy".format(GNN_MODEL))
    threads: 4 if ENV == 'local' else 20
    params:
        BATCH_SIZE= 128,
        NODE_TO_VEC_DIM=16,
        NODE_TO_VEC_EPOCHS=5,
        NUM_WORKERS=4 if ENV == 'local' else 16,
        SET_DEVICE="cuda:0" if GNN_MODEL == 'gat' else "cuda:1"
    run:
        os.environ["SET_GPU"] = params.SET_DEVICE
        import numpy as np
        import torch
        from models.weighted_node2vec import WeightedNode2Vec
        from dataset import triplet_dataset, pokec_data
        from utils.config import DEVICE
        from utils.link_prediction import GATLinkPrediction, GCNLinkPrediction
        import residual2vec as rv
        import warnings
        import gc
        from tqdm import tqdm, trange

        warnings.filterwarnings("ignore")
        gc.enable()

        window_length = 5
        num_walks = 10
        dim = 128
        walk_length = 5

        d = pokec_data.PokecDataFrame()
        edge_index, num_nodes = d.edge_index, d.X.shape[0]
        node_to_vec = WeightedNode2Vec(
            num_nodes=num_nodes,
            group_membership=d.get_grouped_col(),
            weighted_adj=str(input.weighted_adj),
            edge_index=edge_index,
            embedding_dim=params.NODE_TO_VEC_DIM,
            walk_length=walk_length,
            context_size=2,
        ).to(DEVICE)
        model = rv.residual2vec_sgd(
            noise_sampler=rv.ConfigModelNodeSampler(),
            window_length=window_length,
            num_walks=num_walks,
            walk_length=walk_length
        ).fit()
        node_to_vec.load_state_dict(torch.load(str(input.node2vec_weights)))
        X = node_to_vec.embedding.weight.detach().cpu()
        d = triplet_dataset.TripletGraphDataset(X=X, edge_index=edge_index, )
        dataloader = triplet_dataset.NeighborEdgeSampler(d, batch_size=params.BATCH_SIZE, shuffle=False,
            num_workers=params.NUM_WORKERS, pin_memory=True, transforming=False)
        if GNN_MODEL == 'gat':
            m = GATLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=NODE_TO_VEC_DIM)
        elif GNN_MODEL == 'gcn':
            m = GCNLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=NODE_TO_VEC_DIM)
        else:
            raise ValueError("GNN_MODEL must be either gat or gcn")

        # model.transform(model=m, dataloader=dataloader)
        torch.save(m.state_dict(), str(model_weights))

#
# rule generate_embs:
#     # snakemake -R --until generate_embs_crosswalk  -call --config env=local model=gat --nolock
#     input:
#         model_weights=j(DATA_ROOT,"pokec_{}.h5".format(GNN_MODEL)),
#     output:
#         embs_file = j(DATA_ROOT, "pokec_{}_embs_fixed.npy".format(GNN_MODEL))
#     threads: 4 if ENV == 'local' else 20
#     params:
#         BATCH_SIZE= 128,
#         NUM_WORKERS=4 if ENV == 'local' else 16,
#         SET_DEVICE="cuda:0" if GNN_MODEL == 'gat' else "cuda:1"
#     run:
#         os.environ["SET_GPU"] = params.SET_DEVICE
#         import numpy as np
#         import torch
#         from dataset import triplet_dataset, pokec_data
#         from utils.config import DEVICE
#         from utils.link_prediction import GATLinkPrediction, GCNLinkPrediction
#         import residual2vec as rv
#         import warnings
#         import gc
#         from tqdm import tqdm, trange
#
#         warnings.filterwarnings("ignore")
#         gc.enable()
#
#         window_length = 5
#         num_walks = 10
#         dim = 128
#         walk_length = 5
#
#         d = pokec_data.PokecDataFrame()
#         edge_index, num_nodes = d.edge_index, d.X.shape[0]
#
#         model = rv.residual2vec_sgd(
#             noise_sampler=rv.ConfigModelNodeSampler(),
#             window_length=window_length,
#             num_walks=num_walks,
#             walk_length=walk_length
#         ).fit()
#         X = d.X
#         d = triplet_dataset.TripletGraphDataset(X=X, edge_index=edge_index, )
#         dataloader = triplet_dataset.NeighborEdgeSampler(d, batch_size=params.BATCH_SIZE, shuffle=False,
#             num_workers=params.NUM_WORKERS, pin_memory=True, transforming=False)
#         if GNN_MODEL == 'gat':
#             m = GATLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=d.num_features)
#         elif GNN_MODEL == 'gcn':
#             m = GCNLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=d.num_features)
#         else:
#             raise ValueError("GNN_MODEL must be either gat or gcn")
#         m.load_state_dict(torch.load(str(input.model_weights)))
#         m = m.to(DEVICE)
#         m.eval()
#
#         embs = torch.zeros((num_nodes, 128 * 3))
#         with torch.no_grad():
#             for idx, batch in enumerate(tqdm(dataloader,desc="Generating node embeddings")):
#                 a, p, n = batch
#                 a, p, n = m.forward_i(a), m.forward_o(p), m.forward_o(n)
#                 a, p, n = a.detach().cpu(), p.detach().cpu(), n.detach().cpu()
#                 embs[idx * params.BATCH_SIZE:(idx + 1) * params.BATCH_SIZE, :] = torch.cat((a, p, n),dim=1)
#         np.save(str(output.embs_file),embs.numpy())
#
# rule generate_embs_crosswalk:
#     # snakemake -R --until generate_embs_crosswalk  -call --config env=local model=gat --nolock
#     input:
#         model_weights=j(DATA_ROOT,"pokec_crosswalk_{}_nodevec.h5".format(GNN_MODEL)),
#         node2vec_weights=j(DATA_ROOT,"pokec_crosswalk_{}_node2vec.h5".format(GNN_MODEL)),
#         weighted_adj= j(DATA_ROOT,"pokec_crosswalk_adj.npz"),
#     output:
#         embs_file = j(DATA_ROOT, "pokec_crosswalk_{}_embs_fixed.npy".format(GNN_MODEL))
#     threads: 4 if ENV == 'local' else 20
#     params:
#         BATCH_SIZE= 128,
#         NODE_TO_VEC_DIM=16,
#         NODE_TO_VEC_EPOCHS=5,
#         NUM_WORKERS=4 if ENV == 'local' else 16,
#         SET_DEVICE="cuda:0" if GNN_MODEL == 'gat' else "cuda:1"
#     run:
#         os.environ["SET_GPU"] = params.SET_DEVICE
#         import numpy as np
#         import torch
#         from models.weighted_node2vec import WeightedNode2Vec
#         from dataset import triplet_dataset, pokec_data
#         from utils.config import DEVICE
#         from utils.link_prediction import GATLinkPrediction, GCNLinkPrediction
#         import residual2vec as rv
#         import warnings
#         import gc
#         from tqdm import tqdm, trange
#
#         warnings.filterwarnings("ignore")
#         gc.enable()
#
#         window_length = 5
#         num_walks = 10
#         dim = 128
#         walk_length = 5
#
#         d = pokec_data.PokecDataFrame()
#         edge_index, num_nodes = d.edge_index, d.X.shape[0]
#         node_to_vec = WeightedNode2Vec(
#             num_nodes=num_nodes,
#             group_membership=d.get_grouped_col(),
#             weighted_adj=str(input.weighted_adj),
#             edge_index=edge_index,
#             embedding_dim=params.NODE_TO_VEC_DIM,
#             walk_length=walk_length,
#             context_size=2,
#         ).to(DEVICE)
#         model = rv.residual2vec_sgd(
#             noise_sampler=rv.ConfigModelNodeSampler(),
#             window_length=window_length,
#             num_walks=num_walks,
#             walk_length=walk_length
#         ).fit()
#         node_to_vec.load_state_dict(torch.load(str(input.node2vec_weights)))
#         X = node_to_vec.embedding.weight.detach().cpu()
#         d = triplet_dataset.TripletGraphDataset(X=X, edge_index=edge_index, )
#         dataloader = triplet_dataset.NeighborEdgeSampler(d, batch_size=params.BATCH_SIZE, shuffle=False,
#             num_workers=params.NUM_WORKERS, pin_memory=True, transforming=False)
#         if GNN_MODEL == 'gat':
#             m = GATLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=params.NODE_TO_VEC_DIM)
#         elif GNN_MODEL == 'gcn':
#             m = GCNLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=params.NODE_TO_VEC_DIM)
#         else:
#             raise ValueError("GNN_MODEL must be either gat or gcn")
#         m.load_state_dict(torch.load(str(input.model_weights)))
#         m = m.to(DEVICE)
#         m.eval()
#
#         embs = torch.zeros((num_nodes, 128 * 3))
#         with torch.no_grad():
#             for idx, batch in enumerate(tqdm(dataloader,desc="Generating node embeddings")):
#                 a, p, n = batch
#                 a, p, n = m.forward_i(a), m.forward_o(p), m.forward_o(n)
#                 a, p, n = a.detach().cpu(), p.detach().cpu(), n.detach().cpu()
#                 embs[idx * params.BATCH_SIZE:(idx + 1) * params.BATCH_SIZE, :] = torch.cat((a, p, n),dim=1)
#         np.save(str(output.embs_file),embs.numpy())

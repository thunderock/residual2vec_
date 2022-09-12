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
    threads: 20
    params:
        BATCH_SIZE = 128,
        NODE_TO_VEC_DIM = 16,
        NODE_TO_VEC_EPOCHS = 5,
        NUM_WORKERS = 16,
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


rule train_gnn_with_nodevec_unweighted_baseline_generate_embs:
    # snakemake -R --until train_gnn_with_nodevec_unweighted_baseline  -call --config env=local model=gat
    input:
        model_weights = j(DATA_ROOT, "pokec_{}_nodevec.h5".format(GNN_MODEL)),
        node2vec_weights = j(DATA_ROOT, "pokec_{}_node2vec.h5".format(GNN_MODEL))
    output:
        embs_file = j(DATA_ROOT, "pokec_{}_node2vec_embs.npy".format(GNN_MODEL))
    threads: 20
    params:
        BATCH_SIZE = 128,
        NODE_TO_VEC_DIM = 16,
        NODE_TO_VEC_EPOCHS = 5,
        NUM_WORKERS = 16,
        SET_DEVICE = "cuda:0" if (ENV == 'local' or GNN_MODEL == "gat") else "cuda:1"
    run:
        os.environ["SET_GPU"] = params.SET_DEVICE
        import torch
        import numpy as np
        from tqdm import tqdm
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
        node_to_vec.load_state_dict(torch.load(str(input.node2vec_weights), map_location=DEVICE))
        model = rv.residual2vec_sgd(
            noise_sampler=False,
            window_length=window_length,
            num_walks=num_walks,
            walk_length=walk_length
        ).fit()
        X = node_to_vec.embedding.weight.detach().cpu()
        X = torch.cat([X, d.X], dim=1)
        d = triplet_dataset.TripletGraphDataset(X=X, edge_index=edge_index,)
        dataloader = triplet_dataset.NeighborEdgeSampler(d, batch_size=model.batch_size, shuffle=False, num_workers=params.NUM_WORKERS, pin_memory=True)
        if GNN_MODEL == 'gat':
            m = GATLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=params.NODE_TO_VEC_DIM + 5)
        elif GNN_MODEL == 'gcn':
            m = GCNLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=params.NODE_TO_VEC_DIM + 5)
        else:
            raise ValueError("GNN_MODEL must be either gat or gcn")
        m.load_state_dict(torch.load(str(input.model_weights), map_location=DEVICE))
        m = m.to(DEVICE)
        embs = torch.zeros((num_nodes, 128 * 3))
        batch_size = model.batch_size
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataloader,desc="Generating node embeddings")):
                a, p, n = batch
                a, p, n = m.forward_i(a), m.forward_o(p), m.forward_o(n)
                a, p, n = a.detach().cpu(), p.detach().cpu(), n.detach().cpu()
                embs[idx * batch_size:(idx + 1) * batch_size, :] = torch.cat((a, p, n),dim=1)
        np.save(str(output.embs_file),embs.numpy())


rule train_gnn_with_nodevec_crosswalk_baseline:
    # snakemake -R --until train_gnn_with_nodevec_unweighted_baseline  -call --config env=local model=gat
    output:
        model_weights = j(DATA_ROOT, "pokec_crosswalk_{}_nodevec.h5".format(GNN_MODEL)),
        node2vec_weights = j(DATA_ROOT, "pokec_crosswalk_{}_node2vec.h5".format(GNN_MODEL))
    input:
        weighted_adj = j(DATA_ROOT, "pokec_crosswalk_adj.npz"),
    threads: 20
    params:
        BATCH_SIZE = 128,
        NODE_TO_VEC_DIM = 16,
        NODE_TO_VEC_EPOCHS = 5,
        NUM_WORKERS = 16,
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

rule train_gnn_with_nodevec_crosswalk_baseline_generate_embs:
    # snakemake -R --until train_gnn_with_nodevec_unweighted_baseline  -call --config env=local model=gat
    input:
        model_weights = j(DATA_ROOT, "pokec_crosswalk_{}_nodevec.h5".format(GNN_MODEL)),
        node2vec_weights = j(DATA_ROOT, "pokec_crosswalk_{}_node2vec.h5".format(GNN_MODEL)),
        weighted_adj = j(DATA_ROOT, "pokec_crosswalk_adj.npz"),
    output:
        embs_file = j(DATA_ROOT, "pokec_crosswalk_{}_node2vec_embs.npy".format(GNN_MODEL))
    threads: 20
    params:
        BATCH_SIZE = 128,
        NODE_TO_VEC_DIM = 16,
        NODE_TO_VEC_EPOCHS = 5,
        NUM_WORKERS = 16,
        SET_DEVICE = "cuda:0" if GNN_MODEL == "gat" else "cuda:1"
    run:
        os.environ["SET_GPU"] = params.SET_DEVICE
        import torch
        import numpy as np
        from tqdm import tqdm
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
        node_to_vec.load_state_dict(torch.load(str(input.node2vec_weights), map_location=DEVICE))
        model = rv.residual2vec_sgd(
            noise_sampler=False,
            window_length=window_length,
            num_walks=num_walks,
            walk_length=walk_length
        ).fit()

        X = node_to_vec.embedding.weight.detach().cpu()
        X = torch.cat([X, d.X], dim=1)
        d = triplet_dataset.TripletGraphDataset(X=X, edge_index=edge_index,)
        dataloader = triplet_dataset.NeighborEdgeSampler(d, batch_size=model.batch_size, shuffle=False, num_workers=params.NUM_WORKERS, pin_memory=True)
        if GNN_MODEL == 'gat':
            m = GATLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=params.NODE_TO_VEC_DIM + 5)
        elif GNN_MODEL == 'gcn':
            m = GCNLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=params.NODE_TO_VEC_DIM + 5)
        else:
            raise ValueError("GNN_MODEL must be either gat or gcn")
        m.load_state_dict(torch.load(str(input.model_weights), map_location=DEVICE))
        m = m.to(DEVICE)
        embs = torch.zeros((num_nodes, 128 * 3))
        batch_size = model.batch_size
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataloader,desc="Generating node embeddings")):
                a, p, n = batch
                a, p, n = m.forward_i(a), m.forward_o(p), m.forward_o(n)
                a, p, n = a.detach().cpu(), p.detach().cpu(), n.detach().cpu()
                embs[idx * batch_size:(idx + 1) * batch_size, :] = torch.cat((a, p, n),dim=1)
        np.save(str(output.embs_file),embs.numpy())

rule train_gnn_with_nodevec_unweighted_r2v:
    # snakemake -R --until train_gnn_with_nodevec_unweighted_baseline  -call --config env=local model=gat
    output:
        model_weights = j(DATA_ROOT, "pokec_{}_r2v_nodevec.h5".format(GNN_MODEL)),
        node2vec_weights = j(DATA_ROOT, "pokec_{}_r2v_node2vec.h5".format(GNN_MODEL))
    input:
        weighted_adj = j(DATA_ROOT, "pokec_crosswalk_adj.npz"),
    threads: 20
    params:
        BATCH_SIZE = 128,
        NODE_TO_VEC_DIM = 16,
        NODE_TO_VEC_EPOCHS = 5,
        NUM_WORKERS = 16,
        SET_DEVICE = "cuda:0" if GNN_MODEL == "gat" else "cuda:1",
        RV_NUM_WALKS = 100
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
        sbm = triplet_dataset.SbmSamplerWrapper(adj_path=str(input.weighted_adj), group_membership=d.get_grouped_col(),
            window_length=1, padding_id=num_nodes, num_walks=params.RV_NUM_WALKS, num_edges=edge_index.shape[1], use_weights=False)
        edge_index = sbm.edge_index
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
        d = triplet_dataset.TripletGraphDataset(X=X, edge_index=edge_index, sampler=sbm.sample_neg_edges)
        dataloader = triplet_dataset.NeighborEdgeSampler(d, batch_size=model.batch_size, shuffle=True, num_workers=params.NUM_WORKERS, pin_memory=True)
        if GNN_MODEL == 'gat':
            m = GATLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=params.NODE_TO_VEC_DIM + 5)
        elif GNN_MODEL == 'gcn':
            m = GCNLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=params.NODE_TO_VEC_DIM + 5)
        else:
            raise ValueError("GNN_MODEL must be either gat or gcn")

        model.transform(model=m, dataloader=dataloader)
        torch.save(m.state_dict(), str(output.model_weights))

rule train_gnn_with_nodevec_unweighted_r2v_generate_embs:
    # snakemake -R --until train_gnn_with_nodevec_unweighted_baseline  -call --config env=local model=gat
    input:
        model_weights = j(DATA_ROOT, "pokec_{}_r2v_nodevec.h5".format(GNN_MODEL)),
        node2vec_weights = j(DATA_ROOT, "pokec_{}_r2v_node2vec.h5".format(GNN_MODEL)),
        weighted_adj = j(DATA_ROOT, "pokec_crosswalk_adj.npz"),
    output:
        embs_file = j(DATA_ROOT, "pokec_{}_r2v_node2vec_embs.npy".format(GNN_MODEL))
    threads: 20
    params:
        BATCH_SIZE = 128,
        NODE_TO_VEC_DIM = 16,
        NODE_TO_VEC_EPOCHS = 5,
        NUM_WORKERS = 16,
        SET_DEVICE = "cuda:0" if GNN_MODEL == "gat" else "cuda:1",
        RV_NUM_WALKS = 100
    run:
        os.environ["SET_GPU"] = params.SET_DEVICE
        import torch
        import numpy as np
        from tqdm import tqdm
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
        sbm = triplet_dataset.SbmSamplerWrapper(adj_path=str(input.weighted_adj), group_membership=d.get_grouped_col(),
            window_length=1, padding_id=num_nodes, num_walks=params.RV_NUM_WALKS, num_edges=edge_index.shape[1], use_weights=False)
        edge_index = sbm.edge_index
        node_to_vec = UnWeightedNode2Vec(
            num_nodes=num_nodes,
            edge_index=edge_index,
            embedding_dim=params.NODE_TO_VEC_DIM,
            walk_length=walk_length,
            context_size=2,).to(DEVICE)
        node_to_vec.load_state_dict(torch.load(str(input.node2vec_weights), map_location=DEVICE))
        model = rv.residual2vec_sgd(
            noise_sampler=False,
            window_length=window_length,
            num_walks=num_walks,
            walk_length=walk_length
        ).fit()

        X = node_to_vec.embedding.weight.detach().cpu()
        X = torch.cat([X, d.X], dim=1)
        d = triplet_dataset.TripletGraphDataset(X=X, edge_index=edge_index, sampler=sbm.sample_neg_edges)
        dataloader = triplet_dataset.NeighborEdgeSampler(d, batch_size=model.batch_size, shuffle=False, num_workers=params.NUM_WORKERS, pin_memory=True)
        if GNN_MODEL == 'gat':
            m = GATLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=params.NODE_TO_VEC_DIM + 5)
        elif GNN_MODEL == 'gcn':
            m = GCNLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=params.NODE_TO_VEC_DIM + 5)
        else:
            raise ValueError("GNN_MODEL must be either gat or gcn")

        m.load_state_dict(torch.load(str(input.model_weights),map_location=DEVICE))
        m = m.to(DEVICE)
        embs = torch.zeros((num_nodes, 128 * 3))
        batch_size = model.batch_size
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataloader,desc="Generating node embeddings")):
                a, p, n = batch
                a, p, n = m.forward_i(a), m.forward_o(p), m.forward_o(n)
                a, p, n = a.detach().cpu(), p.detach().cpu(), n.detach().cpu()
                embs[idx * batch_size:(idx + 1) * batch_size, :] = torch.cat((a, p, n),dim=1)
        np.save(str(output.embs_file),embs.numpy())

rule train_gnn_with_nodevec_crosswalk_r2v:
    # snakemake -R --until train_gnn_with_nodevec_unweighted_baseline  -call --config env=local model=gat
    output:
        model_weights = j(DATA_ROOT, "pokec_crosswalk_{}_r2v_nodevec.h5".format(GNN_MODEL)),
        node2vec_weights = j(DATA_ROOT, "pokec_crosswalk_{}_r2v_node2vec.h5".format(GNN_MODEL))
    input:
        weighted_adj = j(DATA_ROOT, "pokec_crosswalk_adj.npz"),
    threads: 20
    params:
        BATCH_SIZE = 128,
        NODE_TO_VEC_DIM = 16,
        NODE_TO_VEC_EPOCHS = 5,
        NUM_WORKERS = 16,
        SET_DEVICE = "cuda:0" if GNN_MODEL == "gat" else "cuda:1",
        RV_NUM_WALKS = 100
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
        sbm = triplet_dataset.SbmSamplerWrapper(adj_path=str(input.weighted_adj),group_membership=d.get_grouped_col(),
            window_length=1,padding_id=num_nodes,num_walks=params.RV_NUM_WALKS,num_edges=edge_index.shape[1])
        edge_index = sbm.edge_index

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
        d = triplet_dataset.TripletGraphDataset(X=X,edge_index=edge_index,sampler=sbm.sample_neg_edges)

        dataloader = triplet_dataset.NeighborEdgeSampler(d, batch_size=model.batch_size, shuffle=True, num_workers=params.NUM_WORKERS, pin_memory=True)
        if GNN_MODEL == 'gat':
            m = GATLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=params.NODE_TO_VEC_DIM + 5)
        elif GNN_MODEL == 'gcn':
            m = GCNLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=params.NODE_TO_VEC_DIM + 5)
        else:
            raise ValueError("GNN_MODEL must be either gat or gcn")

        model.transform(model=m, dataloader=dataloader)
        torch.save(m.state_dict(), str(output.model_weights))


rule train_gnn_with_nodevec_crosswalk_r2v_generate_embs:
    # snakemake -R --until train_gnn_with_nodevec_unweighted_baseline  -call --config env=local model=gat
    input:
        model_weights = j(DATA_ROOT, "pokec_crosswalk_{}_r2v_nodevec.h5".format(GNN_MODEL)),
        node2vec_weights = j(DATA_ROOT, "pokec_crosswalk_{}_r2v_node2vec.h5".format(GNN_MODEL)),
        weighted_adj = j(DATA_ROOT, "pokec_crosswalk_adj.npz"),
    output:
        embs_file = j(DATA_ROOT, "pokec_crosswalk_{}_r2v_node2vec_embs.npy".format(GNN_MODEL))
    threads: 20
    params:
        BATCH_SIZE = 128,
        NODE_TO_VEC_DIM = 16,
        NODE_TO_VEC_EPOCHS = 5,
        NUM_WORKERS = 16,
        SET_DEVICE = "cuda:0" if GNN_MODEL == "gat" else "cuda:1",
        RV_NUM_WALKS = 100
    run:
        os.environ["SET_GPU"] = params.SET_DEVICE
        import torch
        import numpy as np
        from tqdm import tqdm
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
        sbm = triplet_dataset.SbmSamplerWrapper(adj_path=str(input.weighted_adj),group_membership=d.get_grouped_col(),
            window_length=1,padding_id=num_nodes,num_walks=params.RV_NUM_WALKS,num_edges=edge_index.shape[1])
        edge_index = sbm.edge_index

        node_to_vec = WeightedNode2Vec(
            num_nodes=num_nodes,
            group_membership=d.get_grouped_col(),
            weighted_adj=str(input.weighted_adj),
            edge_index=edge_index,
            embedding_dim=params.NODE_TO_VEC_DIM,
            walk_length=walk_length,
            context_size=2,).to(DEVICE)
        node_to_vec.load_state_dict(torch.load(str(input.node2vec_weights), map_location=DEVICE))
        model = rv.residual2vec_sgd(
            noise_sampler=False,
            window_length=window_length,
            num_walks=num_walks,
            walk_length=walk_length
        ).fit()
        X = node_to_vec.embedding.weight.detach().cpu()
        X = torch.cat([X, d.X], dim=1)
        d = triplet_dataset.TripletGraphDataset(X=X,edge_index=edge_index,sampler=sbm.sample_neg_edges)

        dataloader = triplet_dataset.NeighborEdgeSampler(d, batch_size=model.batch_size, shuffle=False, num_workers=params.NUM_WORKERS, pin_memory=True)
        if GNN_MODEL == 'gat':
            m = GATLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=params.NODE_TO_VEC_DIM + 5)
        elif GNN_MODEL == 'gcn':
            m = GCNLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=params.NODE_TO_VEC_DIM + 5)
        else:
            raise ValueError("GNN_MODEL must be either gat or gcn")

        m.load_state_dict(torch.load(str(input.model_weights),map_location=DEVICE))
        m = m.to(DEVICE)
        embs = torch.zeros((num_nodes, 128 * 3))
        batch_size = model.batch_size
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataloader,desc="Generating node embeddings")):
                a, p, n = batch
                a, p, n = m.forward_i(a), m.forward_o(p), m.forward_o(n)
                a, p, n = a.detach().cpu(), p.detach().cpu(), n.detach().cpu()
                embs[idx * batch_size:(idx + 1) * batch_size, :] = torch.cat((a, p, n),dim=1)
        np.save(str(output.embs_file),embs.numpy())

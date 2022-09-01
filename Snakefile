from os.path import join as j


rule train_gcn_with_nodevec:
    input:
        weighted_adj = j("/data/sg/ashutiwa/residual2vec_", "pokec_crosswalk_adj.npz"),
    output:
        model_weights = j("/data/sg/ashutiwa/residual2vec_", "pokec_crosswalk_gcn_nodevec.h5"),
    threads: 16
    run:
        import torch
        from models.weighted_node2vec import WeightedNode2Vec
        from dataset import triplet_dataset, pokec_data
        from utils.config import DEVICE
        from tqdm import tqdm, trange
        import gc
        from utils.link_prediction import GCNLinkPrediction
        import residual2vec as rv
        import warnings

        warnings.filterwarnings("ignore")
        gc.enable()

        window_length = 5
        num_walks = 10
        dim = 128
        walk_length = 80
        NUM_WORKERS = 4

        d = pokec_data.PokecDataFrame()
        edge_index, num_nodes = d.edge_index, d.X.shape[0]
        node_to_vec = WeightedNode2Vec(
            num_nodes=num_nodes,
            group_membership=d.get_grouped_col(),
            weighted_adj=str(input.weighted_adj),
            edge_index=edge_index,
            embedding_dim=16,walk_length=walk_length,
            context_size=2,).to(DEVICE)
        k = "degree-unbiased-gcn"
        model = rv.residual2vec_sgd(
            noise_sampler=rv.ConfigModelNodeSampler(),
            window_length=window_length,
            num_walks=num_walks,
            walk_length=walk_length
        ).fit()
        d = triplet_dataset.TripletPokecDataset(node2vec=node_to_vec)
        dataloader = triplet_dataset.NeighborEdgeSampler(d,batch_size=model.batch_size,shuffle=True,num_workers=NUM_WORKERS,pin_memory=True)
        m = GCNLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=d.num_embeddings)
        model.transform(model=m, dataloader=dataloader)
        torch.save(m.state_dict(), str(output.model_weights))


rule train_gat_with_nodevec:
    input:
        weighted_adj = j("/data/sg/ashutiwa/residual2vec_", "pokec_crosswalk_adj.npz"),
    output:
        model_weights = j("/data/sg/ashutiwa/residual2vec_", "pokec_crosswalk_gat_nodevec.h5"),
    threads: 16
    run:
        import torch
        from models.weighted_node2vec import WeightedNode2Vec
        from dataset import triplet_dataset, pokec_data
        from utils.config import DEVICE
        from tqdm import tqdm, trange
        import gc
        from utils.link_prediction import GATLinkPrediction
        import residual2vec as rv
        import warnings

        warnings.filterwarnings("ignore")
        gc.enable()

        window_length = 5
        num_walks = 10
        dim = 128
        walk_length = 80
        NUM_WORKERS = 4

        d = pokec_data.PokecDataFrame()
        edge_index, num_nodes = d.edge_index, d.X.shape[0]
        node_to_vec = WeightedNode2Vec(
            num_nodes=num_nodes,
            group_membership=d.get_grouped_col(),
            weighted_adj=str(input.weighted_adj),
            edge_index=edge_index,
            embedding_dim=16,walk_length=walk_length,
            context_size=2,).to(DEVICE)
        k = "degree-unbiased-gat"
        model = rv.residual2vec_sgd(
            noise_sampler=rv.ConfigModelNodeSampler(),
            window_length=window_length,
            num_walks=num_walks,
            walk_length=walk_length
        ).fit()
        d = triplet_dataset.TripletPokecDataset(node2vec=node_to_vec)
        dataloader = triplet_dataset.NeighborEdgeSampler(d,batch_size=model.batch_size,shuffle=True,num_workers=NUM_WORKERS,pin_memory=True)
        # model = GATLinkPrediction(in_channels=d.num_features,embedding_size=128,num_heads=2,num_layers=5,hidden_channels=64,)
        m = GATLinkPrediction(in_channels=d.num_features, embedding_size=128, num_heads=5, num_layers=5, hidden_channels=64, num_embeddings=d.num_embeddings)
        # m = GCNLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=d.num_embeddings)
        model.transform(model=m, dataloader=dataloader)
        torch.save(m.state_dict(), str(output.model_weights))
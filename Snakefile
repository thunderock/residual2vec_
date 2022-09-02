from os.path import join as j


rule train_gcn_with_nodevec:
    input:
        weighted_adj = j("/data/sg/ashutiwa/residual2vec_", "pokec_crosswalk_adj.npz"),
    output:
        model_weights = j("/data/sg/ashutiwa/residual2vec_", "pokec_crosswalk_gcn_nodevec.h5"),
        node2vec_weights = j("/data/sg/ashutiwa/residual2vec_", "pokec_crosswalk_gcn_node2vec.h5"),
    threads: 16
    params:
        BATCH_SIZE = 128,
        NODE_TO_VEC_DIM = 16,
        NODE_TO_VEC_EPOCHS = 5,
        NUM_WORKERS = 4
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

        d = pokec_data.PokecDataFrame()
        edge_index, num_nodes = d.edge_index, d.X.shape[0]
        node_to_vec = WeightedNode2Vec(
            num_nodes=num_nodes,
            group_membership=d.get_grouped_col(),
            weighted_adj=str(input.weighted_adj),
            edge_index=edge_index,
            embedding_dim=params.NODE_TO_VEC_DIM,walk_length=walk_length,
            context_size=2,).to(DEVICE)
        k = "degree-unbiased-gcn"
        model = rv.residual2vec_sgd(
            noise_sampler=rv.ConfigModelNodeSampler(),
            window_length=window_length,
            num_walks=num_walks,
            walk_length=walk_length
        ).fit()
        loader = node_to_vec.loader(batch_size=params.BATCH_SIZE,
            shuffle=True,
            num_workers=params.NUM_WORKERS,)
        optimizer = torch.optim.Adam(list(node_to_vec.parameters()),lr=0.01)
        X = node_to_vec.train_and_get_embs(loader, optimizer, params.NODE_TO_VEC_EPOCHS, str(output.node2vec_weights))
        d = triplet_dataset.TripletGraphDataset(X=X, edge_index=edge_index,)
        dataloader = triplet_dataset.NeighborEdgeSampler(d, batch_size=model.batch_size, shuffle=True, num_workers=params.NUM_WORKERS, pin_memory=True)
        m = GCNLinkPrediction(in_channels=d.num_features, embedding_size=128, hidden_channels=64,num_layers=5, num_embeddings=params.NODE_TO_VEC_DIM)
        model.transform(model=m, dataloader=dataloader)
        torch.save(m.state_dict(), str(output.model_weights))


rule train_gat_with_nodevec:
    input:
        weighted_adj=j("/data/sg/ashutiwa/residual2vec_","pokec_crosswalk_adj.npz"),
    output:
        model_weights = j("/data/sg/ashutiwa/residual2vec_","pokec_crosswalk_gat_nodevec.h5"),
        node2vec_weights=j("/data/sg/ashutiwa/residual2vec_","pokec_crosswalk_gat_node2vec.h5"),

    threads: 16
    params:
        BATCH_SIZE = 128,
        NODE_TO_VEC_DIM=16,
        NODE_TO_VEC_EPOCHS=5,
        NUM_WORKERS=4
    run:
        import torch
        from models.weighted_node2vec import WeightedNode2Vec
        from dataset import triplet_dataset,pokec_data
        from utils.config import DEVICE
        from tqdm import tqdm,trange
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

        d = pokec_data.PokecDataFrame()
        edge_index,num_nodes=d.edge_index,d.X.shape[0]
        node_to_vec = WeightedNode2Vec(
            num_nodes=num_nodes,
            group_membership=d.get_grouped_col(),
            weighted_adj=str(input.weighted_adj),
            edge_index=edge_index,
            embedding_dim=params.NODE_TO_VEC_DIM,walk_length=walk_length,
            context_size=2,).to(DEVICE)
        k = "degree-unbiased-gat"
        model = rv.residual2vec_sgd(
        noise_sampler=rv.ConfigModelNodeSampler(),
        window_length=window_length,
        num_walks=num_walks,
        walk_length=walk_length
        ).fit()
        loader = node_to_vec.loader(batch_size=params.BATCH_SIZE, shuffle=True, num_workers=params.NUM_WORKERS,)
        optimizer = torch.optim.Adam(list(node_to_vec.parameters()),lr=0.01)
        X = node_to_vec.train_and_get_embs(loader,optimizer,params.NODE_TO_VEC_EPOCHS,str(output.node2vec_weights))
        d = triplet_dataset.TripletGraphDataset(X=X,edge_index=edge_index,)
        dataloader = triplet_dataset.NeighborEdgeSampler(d,batch_size=model.batch_size,shuffle=True,num_workers=params.NUM_WORKERS,pin_memory=True)
        m = GATLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=params.NODE_TO_VEC_DIM)
        model.transform(model=m,dataloader=dataloader)
        torch.save(m.state_dict(),str(output.model_weights))
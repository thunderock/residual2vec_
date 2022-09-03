import os
from os.path import join as j
# os.environ["CUDA_VISIBLE_DEVICES"]=""
# config = {"env": 'local'}
import numpy as np

ENV = config.get('env', 'remote')
DATA_ROOT = "/data/sg/ashutiwa/residual2vec_"
if ENV in ('local', 'carbonate'):
    DATA_ROOT = "data"

rule train_gcn_with_nodevec:
    # snakemake -R --until train_gcn_with_nodevec  -call --config env=local
    input:
        weighted_adj = j(DATA_ROOT, "pokec_crosswalk_adj.npz"),
    output:
        model_weights = j(DATA_ROOT, "pokec_crosswalk_gcn_nodevec.h5"),
        node2vec_weights = j(DATA_ROOT, "pokec_crosswalk_gcn_node2vec.h5"),
    threads: 4 if ENV == 'local' else 20
    params:
        BATCH_SIZE = 128,
        NODE_TO_VEC_DIM = 16,
        NODE_TO_VEC_EPOCHS = 5,
        NUM_WORKERS = 4 if ENV == 'local' else 16,
        SET_DEVICE = "cuda:1"
    run:
        os.environ["SET_GPU"] = params.SET_DEVICE
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
        walk_length = 5

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
        weighted_adj=j(DATA_ROOT,"pokec_crosswalk_adj.npz"),
    output:
        model_weights = j(DATA_ROOT, "pokec_crosswalk_gat_nodevec.h5"),
        node2vec_weights=j(DATA_ROOT, "pokec_crosswalk_gat_node2vec.h5"),

    threads: 4 if ENV == 'local' else 20
    params:
        BATCH_SIZE = 128,
        NODE_TO_VEC_DIM=16,
        NODE_TO_VEC_EPOCHS=5,
        NUM_WORKERS=4 if ENV == 'local' else 16,
        SET_DEVICE= "cuda:0"
    run:
        os.environ["SET_GPU"] = params.SET_DEVICE
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
        walk_length = 5

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

rule train_gcn:
    output:
        model_weights = j(DATA_ROOT, "pokec_gcn.h5")

    threads: 4 if ENV == 'local' else 20
    params:
        NUM_WORKERS=6 if ENV == 'local' else 16,
        SET_DEVICE= "cuda:1"
    run:
        os.environ["SET_GPU"] = params.SET_DEVICE
        import torch
        from dataset import triplet_dataset,pokec_data
        import gc
        from utils.link_prediction import GCNLinkPrediction
        import residual2vec as rv
        import warnings

        warnings.filterwarnings("ignore")
        gc.enable()

        window_length = 5
        num_walks = 10
        dim = 128
        walk_length = 5

        d = pokec_data.PokecDataFrame()
        edge_index,num_nodes=d.edge_index,d.X.shape[0]

        model = rv.residual2vec_sgd(
        noise_sampler=rv.ConfigModelNodeSampler(),
        window_length=window_length,
        num_walks=num_walks,
        walk_length=walk_length
        ).fit()
        X = d.X
        d = triplet_dataset.TripletGraphDataset(X=X,edge_index=edge_index,)
        dataloader = triplet_dataset.NeighborEdgeSampler(d,batch_size=model.batch_size,shuffle=True,num_workers=params.NUM_WORKERS,pin_memory=True)
        # m = GATLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=d.num_features)
        m = GCNLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=d.num_features)

        model.transform(model=m,dataloader=dataloader)
        torch.save(m.state_dict(),str(output.model_weights))

rule train_gat:
    output:
        model_weights = j(DATA_ROOT, "pokec_gat.h5")

    threads: 4 if ENV == 'local' else 20
    params:
        NUM_WORKERS=6 if ENV == 'local' else 16,
        SET_DEVICE= "cuda:0"
    run:
        os.environ["SET_GPU"] = params.SET_DEVICE
        import torch
        from dataset import triplet_dataset,pokec_data
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
        walk_length = 5

        d = pokec_data.PokecDataFrame()
        edge_index,num_nodes=d.edge_index,d.X.shape[0]

        model = rv.residual2vec_sgd(
        noise_sampler=rv.ConfigModelNodeSampler(),
        window_length=window_length,
        num_walks=num_walks,
        walk_length=walk_length
        ).fit()
        X = d.X
        d = triplet_dataset.TripletGraphDataset(X=X,edge_index=edge_index,)
        dataloader = triplet_dataset.NeighborEdgeSampler(d,batch_size=model.batch_size,shuffle=True,num_workers=params.NUM_WORKERS,pin_memory=True)
        m = GATLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,num_embeddings=d.num_features)
        model.transform(model=m,dataloader=dataloader)
        torch.save(m.state_dict(),str(output.model_weights))

rule generate_embs_gat_crosswalk:
    input:
        model_weights=j(DATA_ROOT,"pokec_crosswalk_gat_nodevec.h5"),
        node2vec_weights=j(DATA_ROOT,"pokec_crosswalk_gat_node2vec.h5"),
        weighted_adj= j(DATA_ROOT,"pokec_crosswalk_adj.npz"),
    output:
        embs_file = j(DATA_ROOT, "pokec_crosswalk_gat_embs.h5")
    threads: 4 if ENV == 'local' else 20
    params:
        BATCH_SIZE= 128,
        NODE_TO_VEC_DIM=16,
        NODE_TO_VEC_EPOCHS=5,
        NUM_WORKERS=4 if ENV == 'local' else 16,
        SET_DEVICE="cuda:0"
    run:
        os.environ["SET_GPU"] = params.SET_DEVICE
        import numpy as np
        import torch
        from models.weighted_node2vec import WeightedNode2Vec
        from dataset import triplet_dataset, pokec_data
        from utils.config import DEVICE
        from utils.link_prediction import GATLinkPrediction
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
            num_workers=params.NUM_WORKERS, pin_memory=True, transforming=True)
        m = GATLinkPrediction(in_channels=d.num_features,embedding_size=128,hidden_channels=64,num_layers=5,
            num_embeddings=params.NODE_TO_VEC_DIM)
        m.load_state_dict(torch.load(str(input.model_weights)))
        m = m.to(DEVICE)
        m.eval()
        embs = np.zeros((num_nodes, 128 * 3))
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataloader, desc="Generating node embeddings")):
                a, p, n = batch
                a, p, n = m.forward_i(a), m.forward_o(p), m.forward_o(n)
                a, p, n = a.detach().cpu().numpy(), p.detach().cpu().numpy(), n.detach().cpu().numpy()
                embs[idx * params.BATCH_SIZE:(idx + 1) * params.BATCH_SIZE, :] = np.concatenate((a, p, n), axis=1)
                break
        import pickle as pkl
        pkl.dump(embs, open(str(output.embs_file), "wb"))

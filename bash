snakemake -R --until generate_node_embeddings  -call --config root=data gnn_model=gat env=local device=cuda:0 r2v=true crosswalk=true dataset=airport  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=data gnn_model=gcn env=local device=cuda:0 r2v=true crosswalk=true dataset=airport  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=data gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=true dataset=airport  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=data gnn_model=gcn env=local device=cuda:0 r2v=false crosswalk=true dataset=airport  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=data gnn_model=gat env=local device=cuda:0 r2v=true crosswalk=false dataset=airport  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=data gnn_model=gcn env=local device=cuda:0 r2v=true crosswalk=false dataset=airport  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=data gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=false dataset=airport  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=data gnn_model=gcn env=local device=cuda:0 r2v=false crosswalk=false dataset=airport  --nolock --ignore-incomplete;


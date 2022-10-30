#!/bin/bash
#SBATCH --partition=dl
#SBATCH --time=48:00:00
#SBATCH --job-name=all_fair
#SBATCH --mail-type=ALL
#SBATCH --mem=100G
#SBATCH --mail-user=ashutiwa@iu.edu
#SBATCH --gpus-per-node p100:1
#SBATCH --cpus-per-task=20
#SBATCH -A legacy-projects


which python;
snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gat env=local device=cuda:0 r2v=true crosswalk=false fairwalk=true node2vec=false dataset=polbook  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=false fairwalk=true node2vec=false dataset=polbook  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gcn env=local device=cuda:0 r2v=true crosswalk=false fairwalk=true node2vec=false dataset=polbook  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gcn env=local device=cuda:0 r2v=false crosswalk=false fairwalk=true node2vec=false dataset=polbook  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=word2vec env=local device=cuda:0 r2v=true crosswalk=false fairwalk=true node2vec=false dataset=polbook  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=word2vec env=local device=cuda:0 r2v=false crosswalk=false fairwalk=true node2vec=false dataset=polbook  --nolock --ignore-incomplete;


snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gat env=local device=cuda:0 r2v=true crosswalk=true fairwalk=false node2vec=false dataset=polbook  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=true fairwalk=false node2vec=false dataset=polbook  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gcn env=local device=cuda:0 r2v=true crosswalk=true fairwalk=false node2vec=false dataset=polbook  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gcn env=local device=cuda:0 r2v=false crosswalk=true fairwalk=false node2vec=false dataset=polbook  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=word2vec env=local device=cuda:0 r2v=true crosswalk=true fairwalk=false node2vec=false dataset=polbook  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=word2vec env=local device=cuda:0 r2v=false crosswalk=true fairwalk=false node2vec=false dataset=polbook  --nolock --ignore-incomplete;


snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gat env=local device=cuda:0 r2v=true crosswalk=false fairwalk=false node2vec=false dataset=polbook  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=false fairwalk=false node2vec=false dataset=polbook  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gcn env=local device=cuda:0 r2v=true crosswalk=false fairwalk=false node2vec=false dataset=polbook  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gcn env=local device=cuda:0 r2v=false crosswalk=false fairwalk=false node2vec=false dataset=polbook  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=word2vec env=local device=cuda:0 r2v=true crosswalk=false fairwalk=false node2vec=false dataset=polbook  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=word2vec env=local device=cuda:0 r2v=false crosswalk=false fairwalk=false node2vec=false dataset=polbook  --nolock --ignore-incomplete;



snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gat env=local device=cuda:0 r2v=true crosswalk=false fairwalk=true node2vec=true dataset=polbook  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=false fairwalk=true node2vec=true dataset=polbook  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gcn env=local device=cuda:0 r2v=true crosswalk=false fairwalk=true node2vec=true dataset=polbook  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gcn env=local device=cuda:0 r2v=false crosswalk=false fairwalk=true node2vec=true dataset=polbook  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=word2vec env=local device=cuda:0 r2v=true crosswalk=false fairwalk=true node2vec=true dataset=polbook  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=word2vec env=local device=cuda:0 r2v=false crosswalk=false fairwalk=true node2vec=true dataset=polbook  --nolock --ignore-incomplete;


snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gat env=local device=cuda:0 r2v=true crosswalk=true fairwalk=false node2vec=true dataset=polbook  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=true fairwalk=false node2vec=true dataset=polbook  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gcn env=local device=cuda:0 r2v=true crosswalk=true fairwalk=false node2vec=true dataset=polbook  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gcn env=local device=cuda:0 r2v=false crosswalk=true fairwalk=false node2vec=true dataset=polbook  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=word2vec env=local device=cuda:0 r2v=true crosswalk=true fairwalk=false node2vec=true dataset=polbook  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=word2vec env=local device=cuda:0 r2v=false crosswalk=true fairwalk=false node2vec=true dataset=polbook  --nolock --ignore-incomplete;


snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gat env=local device=cuda:0 r2v=true crosswalk=false fairwalk=false node2vec=true dataset=polbook  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=false fairwalk=false node2vec=true dataset=polbook  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gcn env=local device=cuda:0 r2v=true crosswalk=false fairwalk=false node2vec=true dataset=polbook  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gcn env=local device=cuda:0 r2v=false crosswalk=false fairwalk=false node2vec=true dataset=polbook  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=word2vec env=local device=cuda:0 r2v=true crosswalk=false fairwalk=false node2vec=true dataset=polbook  --nolock --ignore-incomplete;
snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=word2vec env=local device=cuda:0 r2v=false crosswalk=false fairwalk=false node2vec=true dataset=polbook  --nolock --ignore-incomplete;




# snakemake -R --until generate_node_embeddings  -call --config root=polblog gnn_model=gat env=local device=cuda:0 r2v=true crosswalk=true dataset=polblog  --nolock --ignore-incomplete;
# snakemake -R --until generate_node_embeddings  -call --config root=polblog gnn_model=gcn env=local device=cuda:0 r2v=true crosswalk=true dataset=polblog  --nolock --ignore-incomplete;
# snakemake -R --until generate_node_embeddings  -call --config root=polblog gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=true dataset=polblog  --nolock --ignore-incomplete;
# snakemake -R --until generate_node_embeddings  -call --config root=polblog gnn_model=gcn env=local device=cuda:0 r2v=false crosswalk=true dataset=polblog  --nolock --ignore-incomplete;
# snakemake -R --until generate_node_embeddings  -call --config root=polblog gnn_model=gat env=local device=cuda:0 r2v=true crosswalk=false dataset=polblog  --nolock --ignore-incomplete;
# snakemake -R --until generate_node_embeddings  -call --config root=polblog gnn_model=gcn env=local device=cuda:0 r2v=true crosswalk=false dataset=polblog  --nolock --ignore-incomplete;
# snakemake -R --until generate_node_embeddings  -call --config root=polblog gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=false dataset=polblog  --nolock --ignore-incomplete;
# snakemake -R --until generate_node_embeddings  -call --config root=polblog gnn_model=gcn env=local device=cuda:0 r2v=false crosswalk=false dataset=polblog  --nolock --ignore-incomplete;



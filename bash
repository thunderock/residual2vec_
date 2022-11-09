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
# fairwalk + dw
snakemake   train_features_2_vec      -call --config root=polblog_one gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=false fairwalk=true node2vec=false dataset=polblog  --nolock --ignore-incomplete;

# fairwalk node2vec
snakemake   train_features_2_vec      -call --config root=polblog_one gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=false fairwalk=true node2vec=true dataset=polblog  --nolock --ignore-incomplete;

# crosswalk+ dw
snakemake   train_features_2_vec      -call --config root=polblog_one gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=true fairwalk=false node2vec=false dataset=polblog  --nolock --ignore-incomplete;

# crosswalk+ node2vec
snakemake   train_features_2_vec      -call --config root=polblog_one gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=true fairwalk=false node2vec=true dataset=polblog  --nolock --ignore-incomplete;

# r2v + node2vec
snakemake   generate_node_embeddings  -call --config root=polblog_one gnn_model=gat env=local device=cuda:0 r2v=true crosswalk=false fairwalk=false node2vec=true dataset=polblog  --nolock --ignore-incomplete;
snakemake   generate_node_embeddings  -call --config root=polblog_one gnn_model=gcn env=local device=cuda:0 r2v=true crosswalk=false fairwalk=false node2vec=true dataset=polblog  --nolock --ignore-incomplete;

# r2v + dw
snakemake   generate_node_embeddings  -call --config root=polblog_one gnn_model=gat env=local device=cuda:0 r2v=true crosswalk=false fairwalk=false node2vec=false dataset=polblog  --nolock --ignore-incomplete;
snakemake   generate_node_embeddings  -call --config root=polblog_one gnn_model=gcn env=local device=cuda:0 r2v=true crosswalk=false fairwalk=false node2vec=false dataset=polblog  --nolock --ignore-incomplete;


# random + node2vec
snakemake   generate_node_embeddings  -call --config root=polblog_one gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=false fairwalk=false node2vec=true dataset=polblog  --nolock --ignore-incomplete;
snakemake   generate_node_embeddings  -call --config root=polblog_one gnn_model=gcn env=local device=cuda:0 r2v=false crosswalk=false fairwalk=false node2vec=true dataset=polblog  --nolock --ignore-incomplete;

# random + dw
snakemake   generate_node_embeddings  -call --config root=polblog_one gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=false fairwalk=false node2vec=false dataset=polblog  --nolock --ignore-incomplete;
snakemake   generate_node_embeddings  -call --config root=polblog_one gnn_model=gcn env=local device=cuda:0 r2v=false crosswalk=false fairwalk=false node2vec=false dataset=polblog  --nolock --ignore-incomplete;SBATCH -A legacy-projects


which python;
# fairwalk + dw
snakemake   train_features_2_vec      -call --config root=polblog_two gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=false fairwalk=true node2vec=false dataset=polblog  --nolock --ignore-incomplete;

# fairwalk node2vec
snakemake   train_features_2_vec      -call --config root=polblog_two gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=false fairwalk=true node2vec=true dataset=polblog  --nolock --ignore-incomplete;

# crosswalk+ dw
snakemake   train_features_2_vec      -call --config root=polblog_two gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=true fairwalk=false node2vec=false dataset=polblog  --nolock --ignore-incomplete;

# crosswalk+ node2vec
snakemake   train_features_2_vec      -call --config root=polblog_two gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=true fairwalk=false node2vec=true dataset=polblog  --nolock --ignore-incomplete;

# r2v + node2vec
snakemake   generate_node_embeddings  -call --config root=polblog_two gnn_model=gat env=local device=cuda:0 r2v=true crosswalk=false fairwalk=false node2vec=true dataset=polblog  --nolock --ignore-incomplete;
snakemake   generate_node_embeddings  -call --config root=polblog_two gnn_model=gcn env=local device=cuda:0 r2v=true crosswalk=false fairwalk=false node2vec=true dataset=polblog  --nolock --ignore-incomplete;

# r2v + dw
snakemake   generate_node_embeddings  -call --config root=polblog_two gnn_model=gat env=local device=cuda:0 r2v=true crosswalk=false fairwalk=false node2vec=false dataset=polblog  --nolock --ignore-incomplete;
snakemake   generate_node_embeddings  -call --config root=polblog_two gnn_model=gcn env=local device=cuda:0 r2v=true crosswalk=false fairwalk=false node2vec=false dataset=polblog  --nolock --ignore-incomplete;


# random + node2vec
snakemake   generate_node_embeddings  -call --config root=polblog_two gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=false fairwalk=false node2vec=true dataset=polblog  --nolock --ignore-incomplete;
snakemake   generate_node_embeddings  -call --config root=polblog_two gnn_model=gcn env=local device=cuda:0 r2v=false crosswalk=false fairwalk=false node2vec=true dataset=polblog  --nolock --ignore-incomplete;

# random + dw
snakemake   generate_node_embeddings  -call --config root=polblog_two gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=false fairwalk=false node2vec=false dataset=polblog  --nolock --ignore-incomplete;
snakemake   generate_node_embeddings  -call --config root=polblog_two gnn_model=gcn env=local device=cuda:0 r2v=false crosswalk=false fairwalk=false node2vec=false dataset=polblog  --nolock --ignore-incomplete;



which python;
# fairwalk + dw
snakemake   train_features_2_vec      -call --config root=polblog_three gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=false fairwalk=true node2vec=false dataset=polblog  --nolock --ignore-incomplete;

# fairwalk node2vec
snakemake   train_features_2_vec      -call --config root=polblog_three gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=false fairwalk=true node2vec=true dataset=polblog  --nolock --ignore-incomplete;

# crosswalk+ dw
snakemake   train_features_2_vec      -call --config root=polblog_three gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=true fairwalk=false node2vec=false dataset=polblog  --nolock --ignore-incomplete;

# crosswalk+ node2vec
snakemake   train_features_2_vec      -call --config root=polblog_three gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=true fairwalk=false node2vec=true dataset=polblog  --nolock --ignore-incomplete;

# r2v + node2vec
snakemake   generate_node_embeddings  -call --config root=polblog_three gnn_model=gat env=local device=cuda:0 r2v=true crosswalk=false fairwalk=false node2vec=true dataset=polblog  --nolock --ignore-incomplete;
snakemake   generate_node_embeddings  -call --config root=polblog_three gnn_model=gcn env=local device=cuda:0 r2v=true crosswalk=false fairwalk=false node2vec=true dataset=polblog  --nolock --ignore-incomplete;

# r2v + dw
snakemake   generate_node_embeddings  -call --config root=polblog_three gnn_model=gat env=local device=cuda:0 r2v=true crosswalk=false fairwalk=false node2vec=false dataset=polblog  --nolock --ignore-incomplete;
snakemake   generate_node_embeddings  -call --config root=polblog_three gnn_model=gcn env=local device=cuda:0 r2v=true crosswalk=false fairwalk=false node2vec=false dataset=polblog  --nolock --ignore-incomplete;


# random + node2vec
snakemake   generate_node_embeddings  -call --config root=polblog_three gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=false fairwalk=false node2vec=true dataset=polblog  --nolock --ignore-incomplete;
snakemake   generate_node_embeddings  -call --config root=polblog_three gnn_model=gcn env=local device=cuda:0 r2v=false crosswalk=false fairwalk=false node2vec=true dataset=polblog  --nolock --ignore-incomplete;

# random + dw
snakemake   generate_node_embeddings  -call --config root=polblog_three gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=false fairwalk=false node2vec=false dataset=polblog  --nolock --ignore-incomplete;
snakemake   generate_node_embeddings  -call --config root=polblog_three gnn_model=gcn env=local device=cuda:0 r2v=false crosswalk=false fairwalk=false node2vec=false dataset=polblog  --nolock --ignore-incomplete;



which python;
# fairwalk + dw
snakemake   train_features_2_vec      -call --config root=polblog_four gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=false fairwalk=true node2vec=false dataset=polblog  --nolock --ignore-incomplete;

# fairwalk node2vec
snakemake   train_features_2_vec      -call --config root=polblog_four gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=false fairwalk=true node2vec=true dataset=polblog  --nolock --ignore-incomplete;

# crosswalk+ dw
snakemake   train_features_2_vec      -call --config root=polblog_four gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=true fairwalk=false node2vec=false dataset=polblog  --nolock --ignore-incomplete;

# crosswalk+ node2vec
snakemake   train_features_2_vec      -call --config root=polblog_four gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=true fairwalk=false node2vec=true dataset=polblog  --nolock --ignore-incomplete;

# r2v + node2vec
snakemake   generate_node_embeddings  -call --config root=polblog_four gnn_model=gat env=local device=cuda:0 r2v=true crosswalk=false fairwalk=false node2vec=true dataset=polblog  --nolock --ignore-incomplete;
snakemake   generate_node_embeddings  -call --config root=polblog_four gnn_model=gcn env=local device=cuda:0 r2v=true crosswalk=false fairwalk=false node2vec=true dataset=polblog  --nolock --ignore-incomplete;

# r2v + dw
snakemake   generate_node_embeddings  -call --config root=polblog_four gnn_model=gat env=local device=cuda:0 r2v=true crosswalk=false fairwalk=false node2vec=false dataset=polblog  --nolock --ignore-incomplete;
snakemake   generate_node_embeddings  -call --config root=polblog_four gnn_model=gcn env=local device=cuda:0 r2v=true crosswalk=false fairwalk=false node2vec=false dataset=polblog  --nolock --ignore-incomplete;


# random + node2vec
snakemake   generate_node_embeddings  -call --config root=polblog_four gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=false fairwalk=false node2vec=true dataset=polblog  --nolock --ignore-incomplete;
snakemake   generate_node_embeddings  -call --config root=polblog_four gnn_model=gcn env=local device=cuda:0 r2v=false crosswalk=false fairwalk=false node2vec=true dataset=polblog  --nolock --ignore-incomplete;

# random + dw
snakemake   generate_node_embeddings  -call --config root=polblog_four gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=false fairwalk=false node2vec=false dataset=polblog  --nolock --ignore-incomplete;
snakemake   generate_node_embeddings  -call --config root=polblog_four gnn_model=gcn env=local device=cuda:0 r2v=false crosswalk=false fairwalk=false node2vec=false dataset=polblog  --nolock --ignore-incomplete;


which python;
# fairwalk + dw
snakemake   train_features_2_vec      -call --config root=polblog_five gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=false fairwalk=true node2vec=false dataset=polblog  --nolock --ignore-incomplete;

# fairwalk node2vec
snakemake   train_features_2_vec      -call --config root=polblog_five gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=false fairwalk=true node2vec=true dataset=polblog  --nolock --ignore-incomplete;

# crosswalk+ dw
snakemake   train_features_2_vec      -call --config root=polblog_five gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=true fairwalk=false node2vec=false dataset=polblog  --nolock --ignore-incomplete;

# crosswalk+ node2vec
snakemake   train_features_2_vec      -call --config root=polblog_five gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=true fairwalk=false node2vec=true dataset=polblog  --nolock --ignore-incomplete;

# r2v + node2vec
snakemake   generate_node_embeddings  -call --config root=polblog_five gnn_model=gat env=local device=cuda:0 r2v=true crosswalk=false fairwalk=false node2vec=true dataset=polblog  --nolock --ignore-incomplete;
snakemake   generate_node_embeddings  -call --config root=polblog_five gnn_model=gcn env=local device=cuda:0 r2v=true crosswalk=false fairwalk=false node2vec=true dataset=polblog  --nolock --ignore-incomplete;

# r2v + dw
snakemake   generate_node_embeddings  -call --config root=polblog_five gnn_model=gat env=local device=cuda:0 r2v=true crosswalk=false fairwalk=false node2vec=false dataset=polblog  --nolock --ignore-incomplete;
snakemake   generate_node_embeddings  -call --config root=polblog_five gnn_model=gcn env=local device=cuda:0 r2v=true crosswalk=false fairwalk=false node2vec=false dataset=polblog  --nolock --ignore-incomplete;


# random + node2vec
snakemake   generate_node_embeddings  -call --config root=polblog_five gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=false fairwalk=false node2vec=true dataset=polblog  --nolock --ignore-incomplete;
snakemake   generate_node_embeddings  -call --config root=polblog_five gnn_model=gcn env=local device=cuda:0 r2v=false crosswalk=false fairwalk=false node2vec=true dataset=polblog  --nolock --ignore-incomplete;

# random + dw
snakemake   generate_node_embeddings  -call --config root=polblog_five gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=false fairwalk=false node2vec=false dataset=polblog  --nolock --ignore-incomplete;
snakemake   generate_node_embeddings  -call --config root=polblog_five gnn_model=gcn env=local device=cuda:0 r2v=false crosswalk=false fairwalk=false node2vec=false dataset=polblog  --nolock --ignore-incomplete;

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
srun snakemake -R --until generate_node_embeddings  -call --config root=airport gnn_model=gat env=local device=cuda:0 r2v=true crosswalk=true dataset=airport  --nolock --ignore-incomplete;
srun snakemake -R --until generate_node_embeddings  -call --config root=airport gnn_model=gcn env=local device=cuda:0 r2v=true crosswalk=true dataset=airport  --nolock --ignore-incomplete;
srun snakemake -R --until generate_node_embeddings  -call --config root=airport gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=true dataset=airport  --nolock --ignore-incomplete;
srun snakemake -R --until generate_node_embeddings  -call --config root=airport gnn_model=gcn env=local device=cuda:0 r2v=false crosswalk=true dataset=airport  --nolock --ignore-incomplete;
srun snakemake -R --until generate_node_embeddings  -call --config root=airport gnn_model=gat env=local device=cuda:0 r2v=true crosswalk=false dataset=airport  --nolock --ignore-incomplete;
srun snakemake -R --until generate_node_embeddings  -call --config root=airport gnn_model=gcn env=local device=cuda:0 r2v=true crosswalk=false dataset=airport  --nolock --ignore-incomplete;
srun snakemake -R --until generate_node_embeddings  -call --config root=airport gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=false dataset=airport  --nolock --ignore-incomplete;
srun snakemake -R --until generate_node_embeddings  -call --config root=airport gnn_model=gcn env=local device=cuda:0 r2v=false crosswalk=false dataset=airport  --nolock --ignore-incomplete;

srun snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gat env=local device=cuda:0 r2v=true crosswalk=true dataset=polbook  --nolock --ignore-incomplete;
srun snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gcn env=local device=cuda:0 r2v=true crosswalk=true dataset=polbook  --nolock --ignore-incomplete;
srun snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=true dataset=polbook  --nolock --ignore-incomplete;
srun snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gcn env=local device=cuda:0 r2v=false crosswalk=true dataset=polbook  --nolock --ignore-incomplete;
srun snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gat env=local device=cuda:0 r2v=true crosswalk=false dataset=polbook  --nolock --ignore-incomplete;
srun snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gcn env=local device=cuda:0 r2v=true crosswalk=false dataset=polbook  --nolock --ignore-incomplete;
srun snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=false dataset=polbook  --nolock --ignore-incomplete;
srun snakemake -R --until generate_node_embeddings  -call --config root=polbook gnn_model=gcn env=local device=cuda:0 r2v=false crosswalk=false dataset=polbook  --nolock --ignore-incomplete;

srun snakemake -R --until generate_node_embeddings  -call --config root=polblog gnn_model=gat env=local device=cuda:0 r2v=true crosswalk=true dataset=polblog  --nolock --ignore-incomplete;
srun snakemake -R --until generate_node_embeddings  -call --config root=polblog gnn_model=gcn env=local device=cuda:0 r2v=true crosswalk=true dataset=polblog  --nolock --ignore-incomplete;
srun snakemake -R --until generate_node_embeddings  -call --config root=polblog gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=true dataset=polblog  --nolock --ignore-incomplete;
srun snakemake -R --until generate_node_embeddings  -call --config root=polblog gnn_model=gcn env=local device=cuda:0 r2v=false crosswalk=true dataset=polblog  --nolock --ignore-incomplete;
srun snakemake -R --until generate_node_embeddings  -call --config root=polblog gnn_model=gat env=local device=cuda:0 r2v=true crosswalk=false dataset=polblog  --nolock --ignore-incomplete;
srun snakemake -R --until generate_node_embeddings  -call --config root=polblog gnn_model=gcn env=local device=cuda:0 r2v=true crosswalk=false dataset=polblog  --nolock --ignore-incomplete;
srun snakemake -R --until generate_node_embeddings  -call --config root=polblog gnn_model=gat env=local device=cuda:0 r2v=false crosswalk=false dataset=polblog  --nolock --ignore-incomplete;
srun snakemake -R --until generate_node_embeddings  -call --config root=polblog gnn_model=gcn env=local device=cuda:0 r2v=false crosswalk=false dataset=polblog  --nolock --ignore-incomplete;



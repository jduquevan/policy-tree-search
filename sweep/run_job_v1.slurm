#!/bin/bash

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=pts

# Remove one # to uncommment
#SBATCH --output=/network/scratch/j/juan.duque/slurm_output/slurm-%j.out
#SBATCH --error=/network/scratch/j/juan.duque/slurm_output/job-%j.out

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --mem=40G
#SBATCH --time=0-00:29:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=long


# Submit jobs.
module purge
eval "$(conda shell.bash hook)"
conda activate pts
export WANDB_ENTITY="jduque"
export HYDRA_FULL_ERROR=1

python main.py \
    seed=${1} \
    learning_rate=${2} \
    tree_lr=${2} \
    br_entropy_beta=${4} \
    agent_entropy_beta=${4} \
    max_tree_depth=${6} \
    num_br_updates=${7} \
    mlp.features=${8} \
    gru.hidden_dim=${8} \
    output_mlp.features=${8} \
    do_self_play=${9} \
    --config-name='ipd.yaml' \
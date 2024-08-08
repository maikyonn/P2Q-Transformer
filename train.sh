#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --output=output_%j.log           
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=20G
#SBATCH --cpus-per-gpu=64
#SBATCH --partition=gpu

module load conda
conda env create -f environment.yml
conda activate p2q
wandb login b3a1b8c1b6b4c2f38ba6f9ca4ec1e23a0194e4b4

python3 train.py

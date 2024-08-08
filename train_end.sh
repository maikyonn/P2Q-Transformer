#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --output=endeavour_output_%j.log           
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem-per-gpu=100G
#SBATCH --cpus-per-gpu=10
#SBATCH --partition=isi

module load git
source /home1/spangher/miniconda3/etc/profile.d/conda.sh
source activate /home1/spangher/.conda/envs/p2q
pip install -r requirements.txt
wandb login b3a1b8c1b6b4c2f38ba6f9ca4ec1e23a0194e4b4

accelerate launch --multi_gpu --mixed_precision=bf16 train.py 

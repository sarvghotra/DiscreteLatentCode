#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=128G
#SBATCH -c 6
#SBATCH --nodes=1
#SBATCH -p short-unkillable
#SBATCH --time=3:00:00

python train.py --config-name=config model=medium_L512 training.accum=2 training.batch_size=512

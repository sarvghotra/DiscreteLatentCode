#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=64G
#SBATCH -c 6
#SBATCH --nodes=1
#SBATCH -p short-unkillable

GPUS_PER_NODE=4

ulimit -n 16384

accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train_diffusion.py --model DiTSEM-XL/2 --feature-path $SCRATCH/datasets/in_features_dino_vitl_sem_L512 --results-dir $SCRATCH/experiments/test --epochs 1400 --use-sem --ckpt-every 5 --L 512 --V 256

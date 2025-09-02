#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --nodes=1
#SBATCH -p short-unkillable

GPUS_PER_NODE=4

ulimit -n 16384

accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train_diffusion.py --model DiTSEM-XL/2 --hf-dataset lavoies/DLC_512x256 --results-dir $SCRATCH/experiments/test --epochs 100 --use-sem --ckpt-every 5 --L 512 --V 256

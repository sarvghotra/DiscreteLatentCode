#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -C ampere|lovelace


SEED=${SLURM_ARRAY_TASK_ID:=0}

echo $SEED

# source mila.sh
accelerate launch --num_processes 1 --mixed_precision bf16 run_sample.py  --batch_size 256 --total_samples 50_000 --global_seed=$SEED $@

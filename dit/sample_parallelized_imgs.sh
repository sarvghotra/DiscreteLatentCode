#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -C ampere|lovelace


# SEED=${SLURM_ARRAY_TASK_ID:=0}


python sample_ddp_parallelized_dlc.py --task-id ${SLURM_ARRAY_TASK_ID:=0} $@ # --model DiTSEM-XL/2 --image-size 256 --ckpt $SCRATCH/experiments/dit_dino_L512/checkpoints/checkpoints/checkpoint_1106/custom_checkpoint_0.pkl  --sem-dir $SCRATCH/exp/SEM_L512_V256-medium/generated_0/eplast_4096_corrector_eta001/diffused_SEMs --cfg-scale 1.5 --sample-dir $SCRATCH/experiments/dit_dino_L512/manual_eval/eta001_T4096_medium_last --per-proc-batch-size 1 --dino-cfg $SCRATCH/experiments/dinov2-vitl14-pos-shared-sem-L512/config.yaml --num-fid-samples 50_000

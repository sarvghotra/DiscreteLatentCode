#!/bin/bash

export HF_HOME=$SCRATCH/hf_home

cd ..

DLC_DIR=$SCRATCH/reproduce/sedd/it1024
OUT_DIR=$SCRATCH/reproduce/dit/it1024

cd sedd
jid=$(sbatch --parsable --array=0-25 ./sample_sem.sh --steps 1024 --sample_dir $DLC_DIR --model_path lavoies/DLC_SEDD_L512)

cd ../dit
sbatch --dependency=${jid} --array=0-195 ./sample_parallelized_imgs_hf.sh --hf-model lavoies/DLC_DiT_L512 --dlc-dir $DLC_DIR/diffused_SEMs --cfg-scale 1.7  --sample-dir $OUT_DIR

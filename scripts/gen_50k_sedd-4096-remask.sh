#!/bin/bash

export HF_HOME=$SCRATCH/hf_home

cd ..

DLC_DIR=$SCRATCH/reproduce/sedd/it4096_remask-0.01
OUT_DIR=$SCRATCH/reproduce/dit/it4096_remask-0.01

cd sedd
jid=$(sbatch --parsable --array=0-50 --time=20:00:00 ./sample_sem.sh --steps 4096 --sample_dir $DLC_DIR --model_path lavoies/DLC_SEDD_L512 --sample_name corrector --eta=0.01)

cd ../dit
sbatch --dependency=${jid} --array=0-195 --time=1:00:00 ./sample_parallelized_imgs_hf.sh --hf-model lavoies/DLC_DiT_L512 --dlc-dir $DLC_DIR/diffused_SEMs --cfg-scale 1.0  --sample-dir $OUT_DIR

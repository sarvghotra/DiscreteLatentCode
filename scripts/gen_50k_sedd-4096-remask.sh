#!/bin/bash

export HF_HOME=$SCRATCH/hf_home

cd ..

DLC_DIR=$SCRATCH/reproduce/sedd/it4096_remask-0.0005_0.5-0.75
OUT_DIR=$SCRATCH/reproduce/dit/it4096_remask-0.0005_0.5-0.75

cd sedd
jid=$(sbatch --parsable --array=0-25 ./sample_sem.sh --steps 4096 --sample_dir $DLC_DIR --model_path lavoies/DLC_SEDD_L512 --sample_name corrector --eta=0.0005 --t0 0.5 --t1 0.75)

cd ../dit
sbatch --dependency=${jid} --array=0-195 ./sample_parallelized_imgs_hf.sh --hf-model lavoies/DLC_DiT_L512 --dlc-dir $DLC_DIR/diffused_SEMs --cfg-scale 1.0  --sample-dir $OUT_DIR

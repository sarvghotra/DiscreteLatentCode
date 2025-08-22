#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=164G
#SBATCH -c 8
#SBATCH -p short-unkillable

source mila.sh
export WANDB_PROJECT=sem
export HF_HOME=$SCRATCH/hf_home

ulimit -n 8192

# python \
grad_acc_steps=8 # Gradient accumulation seems broken for streamed dataset...
# accelerate launch --main_process_port 29501 --multi_gpu --num_processes 4  --config_file deepspeed_zero2.yaml --mixed_precision bf16 \
# accelerate launch --main_process_port 29501 --multi_gpu --num_processes 2 --config_file deepspeed_zero2.yaml --mixed_precision bf16 \
accelerate launch --main_process_port 29501 --num_processes 4 --config_file deepspeed_zero2.yaml --mixed_precision bf16 \
    --gradient_accumulation_steps $grad_acc_steps \
    train_llada_sem_laion.py \
    --dataset_name /network/scratch/n/noukhovm/experiments/coco-dinov2-vitl14-pos-shared-sem-dataset-temp0.1 \
    --dataset_train_split train \
    --dataset_test_split validation \
    --torch_compile True \
    --output_dir $SCRATCH/experiments/llada_large_sharedsem_laion_L512_v2 \
    --model_name_or_path GSAI-ML/LLaDA-8B-Base \
    --ignore_data_skip True \
    --encoder_path /network/scratch/l/lavoiems/experiments/dinov2-vitl14-sem-L512/eval/training_124999/teacher_checkpoint.pth \
    --L 512 \
    --V 256 \
    --max_seq_length 640 \
    --model_name diffusion \
    --model_type diffusion \
    --learning_rate 1e-5 \
    --warmup_steps 5 \
    --trust_remote_code True \
    --ddp_find_unused_parameters True \
    --dispatch_batches False \
    --lr_scheduler_type cosine \
    --report_to wandb \
    --save_total_limit 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --max_steps 600_000 \
    --gradient_accumulation_steps $grad_acc_steps \
    --torch_dtype bfloat16 \
    --bf16 \
    --tf32 True \
    --logging_steps 200 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 8 \
    --save_steps 0.001 \
    --run_name llada-sem-shared \
    --num_train_epochs 1 $@
 
    # --model_name_or_path k-l-lambda/Llama-3.2-1B-vocab32k \
    # --tf32 True \
    # --save_steps 0.003 \

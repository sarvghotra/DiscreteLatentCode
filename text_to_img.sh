#!/bin/bash

PROMPT="An image of a golden retriever"

python dit/chat_sem.py --model_name_or_path lavoies/DLC_LLADA_L512 --output_path test.pt --remasking random --L 512 --V 256 --temperature 0.2 --steps 512 --num_samples 3 --prompt="$PROMPT"
python dit/sample_sem.py --model lavoies/DLC_DiT_L512 --cfg-scale 3 --image-size 256 --sem-path test.pt

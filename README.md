# Compositional Discrete Latent Code for High Fidelity, Productive Diffusion Models

📄[Paper]() 📁[DLC dataset](#-DLC-datasets) 📀[Models](#-Pre-trained-models) 🖌[BibTex](#-Citation) 📚[References](#-Reference)

**Authors**: Samuel Lavoie, Michael Noukhovitch, Aaron Courville

![Head image -- unconditional and semantic compositional generation examples](figures/head_github.png)

We introduce compositional discrete latent codes (DLCs), which enable both high-fidelity image generation and compositional generation in diffusion models.
This repository contains the official code, DLC datasets, and models for the paper *Compositional Discrete Latent Code for High Fidelity, Productive Diffusion Models.*
Below, we provide the DLC datasets and instructions to load the pre-trained models using HuggingFace 🤗.
The code to reproduce the model is organized as follows:
* The code to reproduce the SEM encoder: [./dinov2](./dinov2)
* The code to reproduce the DLC generator: [./sedd](./sedd)
* The code to reproduce the DLC to image generator and text-and-image LLADA fine-tuning: [./dit](./dit).

## ⚙️  Installation
The python packages to train all of the models can be installed using the following commands:
```bash
virtualenv env
source env/bin/activate
pip install -r requirement.txt
pip install flash_attn
pip install --no-build-isolation --no-deps git+https://github.com/facebookresearch/xformers.git
pip install -e dinov2
```

# 📁 DLC datasets

We provide the ImageNet DLC as a HiggingFace dataset that is produced using the SEM-DinoV2 models.
This dataset is used to train the DLC-SEDD and the DLC-DiT models.

| DLC shape        | HF dataset |
| --------------   | ------- |
| $512\times 256$  | [lavoies/DLC_512x256](https://huggingface.co/datasets/lavoies/DLC_512x256)  |

Using this dataset:
```
from datasets import load_dataset

dataset = load_dataset("lavoies/DLC_512x256", split="train")
features = dataset[0]['features']
dlc = dataset[0]['labels']
```

# 📀 Pre-trained models

We provide pre-trained SEM encoders as HF model and the IN-1k linear probe accuracy on the DLC, which are discretized SEMs.
They are Dinov2 encoders fine-tuned with ImageNet-1k data.
The code to reproduce the encoders can be found in the folder [dinov2](./dinov2).

## Pre-trained SEM Encoders
| DLC shape        | IN1k Lin prob acc on DLC |   HF model    |
| --------------   | ----------------- | ------------- |
| $512\times 256$  | 85.3              | [lavoies/SEM_dinov2_L512](https://huggingface.co/lavoies/SEM_dinov2_L512)  |

Encoding an image using the Huggingface model can be achieved as follows:
```
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained('lavoies/SEM_dinov2_L512', trust_remote_code=True)
model = AutoModel.from_pretrained('lavoies/SEM_dinov2_L512', trust_remote_code=True)

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
sem = outputs.sem
dlc = outputs.dlc
```

## Pre-trained DLC-SEDD
| DLC shape         | HF model |
| --------------    | ------------- |
| $512\times 256$   | [lavoies/DLC_SEDD_L512](https://huggingface.co/lavoies/DLC_SEDD_L512)  |

Loading DLC SEDD can be achieved as follows:
```
from transformers import AutoModel

model = AutoModel.from_pretrained('lavoies/DLC_SEDD_L512', trust_remote_code=True)
```

## Pre-trained DLC-DiT
| DLC shape         | HF model |
| --------------    | ------------- |
| $512\times 256$   | [lavoies/DLC_DiT_L512](https://huggingface.co/lavoies/DLC_DiT_L512)  |

Loading DLC DiT custom pipeline can be achieved as follows:
```
from ditpipeline_dlc_dit import DLCDiTPipeline

pipe = DLCDiTPipeline.from_pretrained('lavoies/DLC_DiT_L512', trust_remote_code=True)
```

## Fine-tuned text-and-image LLADA model
| DLC shape | HF model |
| ----------| -------- |
| $512\times 256$ | [lavoies/DLC_LLADA_L512](https://huggingface.co/lavoies/DLC_LLADA_L512) |
Loading LLADA can be achieved as follows
```
from transformers import AutoModel

model = AutoModel.from_pretrained('lavoies/DLC_LLADA_L512', trust_remote_code=True)
```

## Unconditional generation
Unconditional generation can be achieved running the following scripts:
```
python sedd/run_sample.py --sample_dir . --model_path lavoies/DLC_SEDD_L512  --batch_size 32 --steps 512 --total_samples 32 --save_name uncond
python dit/sample_sem.py --model lavoies/DLC_DiT_L512 --cfg-scale 1.5 --image-size 256 --sem-path diffused_SEMs/uncond.pt
```

## Text-to-image generation
Text-to-image generation can be achieved running the following scripts:
```
PROMPT="An image of a golden retriever"

python dit/chat_sem.py --model_name_or_path lavoies/DLC_LLADA_L512 --output_path test.pt --remasking random --L 512 --V 256 --temperature 0.2 --steps 512 --num_samples 3 --prompt="$PROMPT"
python dit/sample_sem.py --model lavoies/DLC_DiT_L512 --cfg-scale 3 --image-size 256 --sem-path test.pt
```


# 🖌 Citation

# 📚 References

Our implementation builds upon three prior works:
* The SEM encoder is built upon [DinoV2](https://github.com/facebookresearch/dinov2).
* The DLC generative model implementation is built upon [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion)
* The DLC-to-image generation is built upon [Fast-DiT](https://github.com/chuanyangjin/fast-DiT), which itself is built upon [DiT](https://github.com/facebookresearch/DiT).


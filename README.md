# Compositional Discrete Latent Code for High Fidelity, Productive Diffusion Models

📄[Paper]() 📁[DLC dataset](#-DLC-datasets) 📀[Models](#-Pre-trained-models) ⚙️ [Installation](#%EF%B8%8F--installation) 🖌[BibTex](#-Citation) 📚[References](#-Reference)

**Authors**: Samuel Lavoie, Michael Noukhovitch, Aaron Courville

This repository contains the official code, DLC datasets, and models for the paper *Compositional Discrete Latent Code for High Fidelity, Productive Diffusion Models.*
We introduce compositional discrete latent codes (DLCs), which enable both high-fidelity image generation and compositional control in diffusion models.

![Head image -- unconditional and semantic compositional generation examples](figures/head_github.png)

We provide our modifications of each of the codebases in this repository
* [DLC-DinoV2](./dinov2)
* [DLC-SEDD](./sedd)
* [DLC-Fast-DiT](./dit).

# 📁 DLC datasets

We provide the ImageNet DLC as a HiggingFace dataset that is produced using the SEM-DinoV2 models.
This dataset is used to train the DLC-SEDD and the DLC-DiT models.

| DLC shape        | HF dataset |
| --------------   | ------- |
| $32\times 4096$  | [download]()  |
| $128\times 1024$ | [download]()  |
| $512\times 256$  | [download]()  |

# 📀 Pre-trained models

We provide pre-trained SEM encoders as HF model and the IN-1k linear probe accuracy on the DLC, which are discretized SEMs.
They are Dinov2 encoders fine-tuned with ImageNet-1k data.
The code to reproduce the encoders can be found in the folder [dinov2](./dinov2).

## Pre-trained SEM Encoders
| DLC shape        | IN1k Lin prob acc |   HF model    |
| --------------   | ----------------- | ------------- |
| $32\times 4096$  | 81.5              | [lavoie/sem-encoder-large-32x4096]()  |
| $128\times 1024$ | 83.6              | [lavoie/sem-encoder-large-128x1024]()  |
| $512\times 256$  | 85.3              | [lavoie/sem-encoder-large-512x256]()  |

Encoding an image using the Huggingface model can be achieved as follows:
```
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained('lavoie/sem-encoder-large-512x256')
model = AutoModel.from_pretrained('lavoie/sem-encoder-large-512x256')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
sems = outputs.last_hidden_state
```

## Pre-trained DLC-SEDD
| DLC shape         | HF model |
| --------------    | ------------- |
| $32\times 4096$   | [lavoie/dlc-sedd-medium-32x4096]()  |
| $128\times 1024$  | [lavoie/dlc-sedd-medium-128x1024]()  |
| $512\times 256$   | [lavoie/dlc-sedd-medium-512x256]()  |

Sampling DLC can be achieved as follows:
```
from transformers import AutoImageProcessor, AutoModel

model = AutoModel.from_pretrained('lavoie/dlc-sedd-medium-512x256')

outputs = model.generate()
dlc = outputs.last_hidden_state
```

## Pre-trained DLC-DiT
| DLC shape         | HF model |
| --------------    | ------------- |
| $32\times 4096$   | [lavoie/dlc-dit-xl2-32x4096]()  |
| $128\times 1024$  | [lavoie/dlc-dit-xl2-128x1024]()  |
| $512\times 256$   | [lavoie/dlc-dit-xl2-512x256]()  |

```
from transformers import AutoImageProcessor, AutoModel

model = AutoModel.from_pretrained('lavoie/dlc-sedd-medium-512x256')
dit = AutoModel.from_pretrained('lavoie/dlc-dit-medium-512x256')

outputs = model.generate()
dlc = outputs.last_hidden_state
image = dit.generate(dlc)
```

# ⚙️  Installation

The python packages to train all of the models can be installed using the following commands:
```bash
virtualenv env
source env/bin/activate
pip install -r requirement.txt
pip install --no-build-isolation --no-deps git+https://github.com/facebookresearch/xformers.git
pip install -e dinov2
```

# 🖌 Citation

# 📚 References

Our implementation builds upon three prior works:
* The SEM encoder is built of the work of [DinoV2](https://github.com/facebookresearch/dinov2).
* The DLC generative model implementation is built upond [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion)
* The DLC-to-image generation is built upon [Fast-DiT](https://github.com/chuanyangjin/fast-DiT), which itself is built upon [DiT](https://github.com/facebookresearch/DiT).


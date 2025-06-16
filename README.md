# Compositional Discrete Latent Code for High Fidelity, Productive Diffusion Models

📄[Paper]() 📁[DLC dataset](#📁-DLC-datasets) ⚙️ [Models](#Pre-trained-models) 📚[BibTex](#Citation)

Authors: Samuel Lavoie, Michael Noukhovitch, Aaron Courville

Compositional discrete latent codes enable high fidelity and productive diffusion models.
This repository provide the training codes to reproduce the results and every artefacts (DLC dataset, models)
to allow researcher to study compositional and productive generative models.
This project builds on three codebases: [DinoV2](https://github.com/facebookresearch/dinov2), [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion) and [Fast-DiT](https://github.com/chuanyangjin/fast-DiT).
We provide our modifications of each of the codebases in this repository [DLC-DinoV2](./dinov2), [DLC-SEDD](./sedd) and [DLC-Fast-DiT](./dit).

# 📁 DLC datasets

We provide the ImageNet DLC as a HiggingFace dataset that is produced using the DLC-DinoV2 models.
This dataset is used to train the DLC-SEDD and the DLC-DiT models.

| DLC shape        | HF dataset |
| --------------   | ------- |
| $32\times 4096$  | [download]()  |
| $128\times 1024$ | [download]()  |
| $512\times 256$  | [download]()  |

# Pre-trained models

We provide pre-trained SEM encoders as HF model and the IN-1k linear probe accuracy on the DLC, which are discretized SEMs.
They are Dinov2 encoders fine-tuned with ImageNet-1k data.
The code to reproduce the encoders can be found in the folder [dinov2](./dinov2).
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
last_hidden_states = outputs.last_hidden_state
```

Pre-trained DLC-SEDD models:
| DLC shape         | HF model |
| --------------    | ------------- |
| $32\times 4096$   | [lavoie/dlc-sedd-medium-32x4096]()  |
| $128\times 1024$  | [lavoie/dlc-sedd-medium-128x1024]()  |
| $512\times 256$   | [lavoie/dlc-sedd-medium-512x256]()  |

Pre-trained DLC-DiT models:
| DLC shape         | HF model |
| --------------    | ------------- |
| $32\times 4096$   | [lavoie/dlc-dit-xl2-32x4096]()  |
| $128\times 1024$  | [lavoie/dlc-dit-xl2-128x1024]()  |
| $512\times 256$   | [lavoie/dlc-dit-xl2-512x256]()  |

# Citation

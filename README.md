# Compositional Discrete Latent Code for High Fidelity, Productive Diffusion Models

[paper]() [DLC dataset]() [Models]() [BibTex]()

Authors: Samuel Lavoie, Michael Noukhovitch, Aaron Courville

Compositional discrete latent codes enable high fidelity and productive diffusion models.
This repository provide the training codes to reproduce the results and every artefacts (DLC dataset, models)
to allow researcher to study compositional and productive generative models.
This project builds on three codebases: [DinoV2](https://github.com/facebookresearch/dinov2), [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion) and [Fast-DiT](https://github.com/chuanyangjin/fast-DiT).
We provide our modifications of each of the codebases in this repository [DLC-DinoV2](), [DLC-SEDD]() and [DLC-Fast-DiT]().

# DLC datasets

We provide the ImageNet DLC as a HiggingFace dataset that is produced using the DLC-DinoV2 models.
This dataset is used to train the DLC-SEDD and the DLC-DiT models.

| DLC shape        | HF dataset |
| --------------   | ------- |
| $32\times 4096$  | [download]()  |
| $128\times 1024$ | [download]()  |
| $512\times 256$  | [download]()  |

# Pre-trained models

Pre-trained DLC-DINOv2 models:
| DLC shape        | IN1k Lin prob acc | HF dataset |
| --------------   | ----------------- | ------------- |
| $32\times 4096$  |                   | [download]()  |
| $128\times 1024$ |                   | [download]()  |
| $512\times 256$  |                   | [download]()  |

Pre-trained DLC-SEDD models:
| DLC shape         | HF dataset |
| --------------    | ------------- |
| $32\times 4096$   | [download]()  |
| $128\times 1024$  | [download]()  |
| $512\times 256$   | [download]()  |

Pre-trained DLC-DiT models:
| DLC shape         | HF dataset |
| --------------    | ------------- |
| $32\times 4096$   | [download]()  |
| $128\times 1024$  | [download]()  |
| $512\times 256$   | [download]()  |


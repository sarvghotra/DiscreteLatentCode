## Scalable Diffusion Models with Transformers (DiT)<br><sub>Improved PyTorch Implementation</sub>


## Training
### Preparation Before Training
To extract ImageNet DLC features with `1` GPUs on one node:

```bash
torchrun --nnodes=1 --nproc_per_node=1 extract_features_withdlc.py --model DiT-XL/2 --data-path /path/to/imagenet/train --features-path /path/to/store/features
```

### Training DiT
We provide a training script for DiT in [`train.py`](train.py). This script can be used to train class-conditional 
DiT models, but it can be easily modified to support other types of conditioning. 

To launch DiT-XL/2 (256x256) training with `1` GPUs on one node:

```bash
accelerate launch --mixed_precision fp16 train_diffusion.py --model DiTSEM-XL/2 --features-path /path/to/store/features
```

To launch DiT-XL/2 (256x256) training with `N` GPUs on one node:
```bash
accelerate launch --multi_gpu --num_processes N --mixed_precision fp16 train.py --model DiTSEM-XL/2 --features-path /path/to/store/features
```

Alternatively, you have the option to extract and train the scripts located in the folder [training options](train_options).


### PyTorch Training Results

We've trained DiT-XL/2 and DiT-B/4 models from scratch with the PyTorch training script
to verify that it reproduces the original JAX results up to several hundred thousand training iterations. Across our experiments, the PyTorch-trained models give 
similar (and sometimes slightly better) results compared to the JAX-trained models up to reasonable random variation. Some data points:

| DiT Model  | Train Steps | FID-50K<br> (JAX Training) | FID-50K<br> (PyTorch Training) | PyTorch Global Training Seed |
|------------|-------------|----------------------------|--------------------------------|------------------------------|
| XL/2       | 400K        | 19.5                       | **18.1**                       | 42                           |
| B/4        | 400K        | **68.4**                   | 68.9                           | 42                           |
| B/4        | 400K        | 68.4                       | **68.3**                       | 100                          |

These models were trained at 256x256 resolution; we used 4x H100s to train XL/2 and 4x A100s to train B/4. Note that FID 
here is computed with 250 DDPM sampling steps, with the `mse` VAE decoder and without guidance (`cfg-scale=1`). 

## Evaluation (FID, Inception Score, etc.)

We include a [`sample_ddp.py`](sample_ddp.py) script which samples a large number of images from a DiT model in parallel. This script 
generates a folder of samples as well as a `.npz` file which can be directly used with [ADM's TensorFlow
evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to compute FID, Inception Score and
other metrics. For example, to sample 50K images from our pre-trained DiT-XL/2 model over `N` GPUs, run:

```bash
torchrun --nnodes=1 --nproc_per_node=N sample_ddp_computed_dlc.py --model DiTSEM-XL/2 --num-fid-samples 50000
```

## Citation

```bibtex
@misc{jin2024fast,
    title={Fast-DiT: Fast Diffusion Models with Transformers},
    author={Jin, Chuanyang and Xie, Saining},
    howpublished = {\url{https://github.com/chuanyangjin/fast-DiT}},
    year={2024}
}
```

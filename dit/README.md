## DLC+DiT -- PyTorch Implementation

## Training
### Preparation Before Training
To extract ImageNet DLC features with `1` GPUs on one node for a $512\times 256$ DLC configuration:
```bash
torchrun --rdzv_endpoint=localhost:29400 --nnodes=1 --nproc_per_node=1 extract_features_withdlc.py --data-path /path/to/imagenet/train --features-path /path/to/store/features --encoder-path /path/to/sem_encoder/teacher_checkpoint.pth --L 512 --V 256 --global-batch-size 512
```

### Training DiT
We provide a training script for DiT in [`train_diffusion.py`](train_diffusion.py). This script can be used to train DLC-conditional 
DiT models. 

See `./train_scripts` for examples of bash invocations to train DLC-DiT models.

## Evaluation (FID, Inception Score, etc.)

We include a [`sample_parallelized_imgs.sh`](sample_parallelized_imgs.sh) script which samples a large number of images from a DiT model and pre-computed
DLC sampled from the SEDD model in parallel. This script 
generates a folder of samples as well as a `.npz` file which can be directly used with [ADM's TensorFlow
evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to compute FID, Inception Score and
other metrics. For example, to sample 50K images from our pre-trained DiTSEM-XL/2 model over `N` GPUs, run:

```bash
sbatch --array=0-195 sample_parallelized_imgs.sh --ckpt /path/to/dit/custom_checkpoint_0.pkl --dino-cfg /path/to/dino/cfg.yaml --dlc-dit /path/to/dlc
```

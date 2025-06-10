# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""

import torch

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
import logging
import os
from collections import OrderedDict
from typing import Any, Tuple

import numpy as np
from accelerate import Accelerator
from diffusers.models import AutoencoderKL
from PIL import Image
from safetensors.torch import load_file
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from vision_transformer import vit_base_sem, vit_large_sem

#################################################################################
#                             Training Helper Functions                         #
#################################################################################


def load_pretrained(model, path):
    if "safetensors" in path:
        checkpoint = load_file(path)
    else:
        checkpoint = torch.load(
            path,
            map_location="cpu",
        )
    if "teacher" in checkpoint:
        state_dict = checkpoint["teacher"]
    else:
        state_dict = checkpoint
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded pre-trained predictor-sem at {path} with msg: {msg}")
    return model


class PathImageSEMFolder(ImageFolder):
    def __init__(self, *args, transform_encoder, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform_encoder = transform_encoder

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample_encoder = self.transform_encoder(sample)
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, sample_encoder, target, path


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{logging_dir}/log.txt"),
        ],
    )
    logger = logging.getLogger(__name__)
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--encoder-path", type=str, default=None)
    parser.add_argument("--features-path", type=str, default="features")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument(
        "--vae", type=str, choices=["ema", "mse"], default="ema"
    )  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--L", type=int, default=4096)
    parser.add_argument("--V", type=int, default=32)
    args = parser.parse_args()
    return args


def parse_args_stage(meta_args, current_iteration):
    args = parse_args()
    feature_path = os.path.join(
        meta_args.experiment_dir, "features", f"iteration_{current_iteration}"
    )
    args.features_path = feature_path
    args.data_path = <IN_PATH>
    encoder_path = os.path.join(
        meta_args.experiment_dir,
        "prediction",
        f"iteration_{current_iteration-1}",
        "checkpoints",
        "checkpoints",
    )
    ckpt_it = max([int(d.split("_")[1]) for d in os.listdir(encoder_path)])
    encoder_path = os.path.join(
        encoder_path, f"checkpoint_{ckpt_it}", "model.safetensors"
    )

    args.encoder_path = encoder_path
    return args


#################################################################################
#                                  Training Loop                                #
#################################################################################


@torch.no_grad()
def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup a feature folder:
    os.makedirs(args.features_path, exist_ok=True)
    os.makedirs(os.path.join(args.features_path, "imagenet256_features"), exist_ok=True)
    os.makedirs(os.path.join(args.features_path, "imagenet256_labels"), exist_ok=True)
    os.makedirs(os.path.join(args.features_path, "imagenet256_path"), exist_ok=True)
    os.makedirs(os.path.join(args.features_path, "imagenet256_sem"), exist_ok=True)

    # Create model:
    assert (
        args.image_size % 8 == 0
    ), "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    accelerator = Accelerator()
    device = accelerator.device
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Setup data:
    transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda pil_image: center_crop_arr(pil_image, args.image_size)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]
    )
    transform_encoder = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = PathImageSEMFolder(
        args.data_path, transform_encoder=transform_encoder, transform=transform
    )
    bsz = 128
    loader = DataLoader(
        dataset,
        batch_size=bsz,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    dino_sem = vit_large_sem(
        img_size=518,
        patch_size=14,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        drop_path_rate=0.3,
        init_values=1.0e-5,
        L=args.L,
        V=args.V,
        drop_path_uniform=True,
    )
    dino_sem = dino_sem.eval()
    dino_sem = dino_sem.cuda()
    dino_sem.sem_out = torch.nn.Identity()
    dino_sem.sem_cls_norm = torch.nn.Identity()
    dino_sem.norm = torch.nn.Identity()
    dino_sem = load_pretrained(dino_sem, args.encoder_path)
    L = dino_sem.L
    V = dino_sem.V

    loader = accelerator.prepare(loader)

    train_steps = 0
    for x, x_enc, y, paths in tqdm(loader):
        x = x.to(device)
        x_enc = x_enc.to(device)
        y = y.to(device)
        with torch.no_grad():
            sem = dino_sem(x_enc)
            sem = sem.view(-1, L, V)
            sem = sem.argmax(-1)
            # Map input images to latent space + normalize latents:
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)

        x = x.detach().cpu().numpy()  # (bsz, 4, 32, 32)
        y = y.detach().cpu().numpy()  # (bsz,)
        sem = sem.detach().cpu().numpy()
        for i, (features, label, s, path) in enumerate(zip(x, y, sem, paths)):
            idx = str(path.split("/")[-1].split(".")[0])
            np.save(f"{args.features_path}/imagenet256_features/{idx}.npy", features)
            np.save(f"{args.features_path}/imagenet256_labels/{idx}.npy", label)
            np.save(f"{args.features_path}/imagenet256_path/{idx}.npy", path)
            np.save(f"{args.features_path}/imagenet256_sem/{idx}.npy", s)

        train_steps += bsz


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    args = parse_args()
    main(args)

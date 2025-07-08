# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
import os

import numpy as np
import yaml
from pipeline_dlc_dit import DLCDiTPipeline
from diffusers.models import AutoencoderKL
from download import find_model
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = DLCDiTPipeline.from_pretrained(args.model, trust_remote_code=True)
    pipe = pipe.to(device)
    dlcs = torch.load(args.sem_path)
    # Setup classifier-free guidance:
    samples = pipe(
        dlcs=dlcs, num_inference_steps=250  # torch.nn.functional.one_hot(y, model.transformer.config.dlc_v)
    ).images

    # Save and display images:
    sem_name = os.path.basename(args.sem_path).split(".")[0]
    save_path = os.path.join("figures", sem_name)
    save_path = os.path.join(
        save_path, f"{args.model.replace('/', '-')}_cfg{args.cfg_scale}.png"
    )
    os.makedirs(save_path, exist_ok=True)
    for i, sample in enumerate(samples):
        sample.save(f"{save_path}/img_{i}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="lavoies/DLC_DiT_L512",
    )
    parser.add_argument("--sem-path", type=str, required=True)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)

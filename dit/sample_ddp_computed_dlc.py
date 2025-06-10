# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch._dynamo.config.optimize_ddp = False

import argparse
import math
import os
import shutil
from datetime import datetime

import numpy as np
import torch
import yaml
from accelerate import Accelerator, InitProcessGroupKwargs
from diffusers.models import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CONFIG_MAPPING, AutoModelForCausalLM

from diffusion import create_diffusion
from download import find_model
from models import DiT_models


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    fns = os.listdir(sample_dir)
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        fn = f"{sample_dir}/{fns[i]}"
        if os.path.exists(fn):
            sample_pil = Image.open(fn)
            sample_np = np.asarray(sample_pil).astype(np.uint8)
            samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


class CustomDataset(Dataset):
    def __init__(self, sem_dir, n_imgs, num_fid_samples):
        self.sem_dir = sem_dir

        sem_files = sorted(os.listdir(sem_dir))
        n_sems_per_file = len(
            torch.load(os.path.join(sem_dir, sem_files[0]), map_location="cpu")
        )
        start_idx = n_imgs // n_sems_per_file
        self.sem_files = sem_files[start_idx:]
        self.start_idx = start_idx
        self.n_sems_per_file = n_sems_per_file
        end_idx = math.ceil(num_fid_samples / n_sems_per_file)

        self.sem_files = sem_files[start_idx:end_idx]

    def __len__(self):
        return len(self.sem_files)

    def __getitem__(self, idx):
        sem_file = self.sem_files[idx]

        sem = torch.load(os.path.join(self.sem_dir, sem_file), map_location="cpu")
        return sem


@torch.inference_mode()
def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = (
        args.tf32
    )  # True: fast but may lead to some small numerical differences
    assert (
        torch.cuda.is_available()
    ), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    pg_kwargs = InitProcessGroupKwargs(timeout=18000)
    accelerator = Accelerator(pg_kwargs)
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}")

    if args.ckpt is None:
        assert (
            args.model == "DiT-XL/2"
        ), "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    with open(args.dino_cfg, "r") as f:
        cfg = yaml.safe_load(f)
        L = cfg["student"]["L"]
        V = cfg["student"]["V"]
    if "SEM" in args.model:
        model = DiT_models[args.model](
            input_size=latent_size, num_classes=args.num_classes, L=L, V=V
        )
    elif "CONT" in args.model:
        model = DiT_models[args.model](
            input_size=latent_size, num_classes=args.num_classes, in_dim=1024
        )
    else:
        model = DiT_models[args.model](
            input_size=latent_size, num_classes=args.num_classes
        )
    model = model.to(device)
    print("Evaluation args", args)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    new_state_dict = dict(state_dict)
    for k, v in state_dict.items():
        if "_orig_mod." in k:
            new_state_dict[k.replace("_orig_mod.", "")] = v
            del new_state_dict[k]
    model.load_state_dict(new_state_dict)
    print("Model", model)
    model = torch.compile(model)

    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = (
        os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    )
    folder_name = (
        f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-"
        f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    )
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    # dist.barrier()
    accelerator.wait_for_everyone()

    n_imgs = len(os.listdir(sample_folder_dir))
    dataset = CustomDataset(args.sem_dir, n_imgs, args.num_fid_samples)

    n = args.per_proc_batch_size
    global_batch_size = n * world_size

    loader = DataLoader(
        dataset,
        batch_size=n,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    loader = accelerator.prepare(loader)

    total = dataset.start_idx * dataset.n_sems_per_file
    for ys in tqdm(loader):
        # FIXME: There seems to be a bug in the dataloader when the size of the dataset is 0
        if ys is None:
            break
        # Sample inputs:
        ys = ys.view(-1, ys.shape[-1])
        for y in tqdm(torch.chunk(ys, 4)):
            y = y.view(-1, y.shape[-1])
            y = y.to(device)
            n = len(y)
            z = torch.randn(
                n, model.in_channels, latent_size, latent_size, device=device
            )
            # Setup classifier-free guidance:
            if using_cfg:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([[V] * model.sem_embedder.L] * n, device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, t_max=args.t_max)
                sample_fn = model.forward_with_cfg
            else:
                model_kwargs = dict(y=y)
                sample_fn = model.forward

            # Sample images:
            samples = diffusion.p_sample_loop(
                sample_fn,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=device,
            )
            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

            samples = vae.decode(samples / 0.18215).sample
            samples = (
                torch.clamp(127.5 * samples + 128.0, 0, 255)
                .permute(0, 2, 3, 1)
                .to("cpu", dtype=torch.uint8)
                .numpy()
            )

            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = i * world_size + rank + total
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
            total += n * world_size
            accelerator.wait_for_everyone()

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    accelerator.wait_for_everyone()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
        shutil.rmtree(sample_folder_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2"
    )
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--dino-cfg", type=str, required=True)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--t-max", type=float, default=None)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).",
    )
    parser.add_argument(
        "--sem-dir",
        type=str,
        default=None,
        help="Path where to find the SEMs to generate",
    )
    args = parser.parse_args()
    main(args)

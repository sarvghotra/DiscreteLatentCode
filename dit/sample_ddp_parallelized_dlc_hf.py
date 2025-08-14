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
from pipeline_dlc_dit import DLCDiTPipeline

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
    def __init__(self, dlc_dir, n_imgs, num_fid_samples):
        self.dlc_dir = dlc_dir

        sem_files = sorted(os.listdir(dlc_dir))
        n_sems_per_file = len(
            torch.load(os.path.join(dlc_dir, sem_files[0]), map_location="cpu")
        )
        self.sem_files = sem_files
        self.n_sems_per_file = n_sems_per_file

    def __len__(self):
        return len(self.sem_files)

    def __getitem__(self, idx):
        sem_file = self.sem_files[idx]

        sem = torch.load(os.path.join(self.dlc_dir, sem_file), map_location="cpu")
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
    seed = args.global_seed + args.task_id
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}")

    pipe = DLCDiTPipeline.from_pretrained(args.hf_model, trust_remote_code=True)
    pipe = pipe.to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.hf_model.replace("/", "-")
    ckpt_string_name = "huggingface" 
    # (
    #     os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "huggingface"
    # )
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

    dataset = CustomDataset(args.dlc_dir, 0, args.num_fid_samples)
    print(f"task index: {args.task_id}, sem_file: {dataset.sem_files[args.task_id]}")
    ys = dataset[args.task_id]

    n = args.per_proc_batch_size
    global_batch_size = n * world_size

    # Sample inputs:
    total = 0
    ys = ys.view(-1, ys.shape[-1])
    if True:
        for y in tqdm(torch.chunk(ys, 4)):
            y = y.view(-1, y.shape[-1])
            y = y.to(device)
            n = len(y)
            samples = pipe(dlcs=y, num_inference_steps=256, guidance_scale=1.0).images

            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = i + args.task_id * len(ys) + total
                sample.save(f"{sample_folder_dir}/{index:06d}.png")
            total += n
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
        "--hf-model", type=str
    )
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--t-max", type=float, default=None)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--total-num", type=int, default=196)
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.",
    )
    parser.add_argument(
        "--dlc-dir",
        type=str,
        default=None,
        help="Path where to find the SEMs to generate",
    )
    args = parser.parse_args()
    main(args)

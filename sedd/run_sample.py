import argparse
import math
import os
import uuid
from datetime import datetime

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm
from transformers import GPT2TokenizerFast

import sampling
from load_model import load_model


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--checkpoint_name", default=None, type=str)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--total_samples", type=int, default=50_000)
    parser.add_argument("--sample_dir", type=str, default=None)
    parser.add_argument("--sample_name", type=str, default="analytic")
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--t0", type=float, default=0.3)
    parser.add_argument("--t1", type=float, default=0.55)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--save_name", type=str, default=None)
    args = parser.parse_args()

    accelerator = Accelerator()
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    device = rank % torch.cuda.device_count()

    # Test compilation
    # Add accelerate multi gpu
    folder_name = "diffused_SEMs"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    accelerator.wait_for_everyone()
    n = args.batch_size
    global_batch_size = n * world_size
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    n_current_samples = len(os.listdir(sample_folder_dir)) * n
    total_samples = int(
        math.ceil((args.total_samples - n_current_samples) / global_batch_size)
        * global_batch_size
    )

    seed = (
        args.global_seed * world_size + rank + n_current_samples
    )  # + datetime.now().timestamp()
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, world_size={world_size}, {seed=}")
    print(f"Running with {args=}")

    model, graph, noise = load_model(args.model_path, "cuda", args.checkpoint_name)
    length = model.pos_embed.shape[0]
    model.forward = torch.compile(model.forward)

    sampling_fn = sampling.get_pc_sampler(
        graph,
        noise,
        (args.batch_size, length),
        args.sample_name,  # "analytic",
        args.steps,
        device="cuda",
        eta=args.eta,
        t0=args.t0,
        t1=args.t1,
    )

    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert (
        total_samples % world_size == 0
    ), "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // world_size)
    assert (
        samples_needed_this_gpu % n == 0
    ), "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar

    for _ in pbar:
        if (len(os.listdir(sample_folder_dir)) * args.batch_size) > args.total_samples:
            exit(0)
        samples = sampling_fn(model)
        if args.save_name is None:
            path = os.path.join(sample_folder_dir, f"{uuid.uuid4()}.pt")
        else:
            path = os.path.join(sample_folder_dir, f"{args.save_name}.pt")
        torch.save(samples.cpu(), path)


if __name__ == "__main__":
    main()

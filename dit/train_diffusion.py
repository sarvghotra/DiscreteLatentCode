"""
A minimal training script for DiT.
"""

import torch

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch._dynamo.config.optimize_ddp = False
import argparse
import json
import logging
import os
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time

import numpy as np
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers.models import AutoencoderKL
from PIL import Image
from safetensors.torch import load_file
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_from_disk, load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from diffusion import create_diffusion
from models import DiT_models

#################################################################################
#                             Training Helper Functions                         #
#################################################################################


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
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


class CustomDataset(Dataset):
    def __init__(self, features_dir, labels_dir):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        assert len(self.features_files) == len(
            self.labels_files
        ), "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]
        assert (
            feature_file == label_file
        ), f"Unexpectedly unmatched feature and label files {feature_file}, {label_file}"

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, default="features")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--hf-dataset", type=str, default=None)
    parser.add_argument(
        "--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2"
    )
    parser.add_argument("--pretrained-path", type=str, default=None)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--exit-after-save", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--L", type=int, default=4096)
    parser.add_argument("--V", type=int, default=32)
    parser.add_argument("--in-dim", type=int, default=1024)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument(
        "--vae", type=str, choices=["ema", "mse"], default="ema"
    )  # Choice doesn't affect training
    parser.add_argument("--use-sem", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=4)
    parser.add_argument("--ckpt-total-limit", type=int, default=None)
    args = parser.parse_args()
    return args


def parse_args_stage(meta_args, current_iteration):
    args = parse_args()
    feature_path = os.path.join(
        meta_args.experiment_dir, "features", f"iteration_{current_iteration}"
    )
    args.feature_path = feature_path
    results_dir = os.path.join(
        meta_args.experiment_dir, "diffusion", f"iteration_{current_iteration}"
    )
    args.results_dir = results_dir
    pretrained_path = os.path.join(
        meta_args.experiment_dir,
        "diffusion",
        f"iteration_{current_iteration-1}",
        "checkpoints",
        "checkpoints",
    )
    ckpt_it = max([int(d.split("_")[1]) for d in os.listdir(pretrained_path)])
    pretrained_path = os.path.join(
        pretrained_path, f"checkpoint_{ckpt_it}", "model.safetensors"
    )
    args.pretrained_path = pretrained_path
    args.use_sem = True
    args.epochs = 40
    args.model = "DiTSEM-XL/2"
    args.ckpt_total_limit = 3
    return args


#################################################################################
#                                  Training Loop                                #
#################################################################################

def collate_tensor_fn(batch):
    # Extend this function to handle batch of tensors
    f = [b['features'] for b in batch]
    l = [b['labels'] for b in batch]
    f = torch.tensor(f)
    l = torch.tensor(l)
    return f, l


def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator:
    os.makedirs(
        args.results_dir, exist_ok=True
    )  # Make results folder (holds all experiment subfolders)
    model_string_name = args.model.replace(
        "/", "-"
    )  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    # Verify that it is the right name.
    experiment_dir = f"{args.results_dir}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")
    project_config = ProjectConfiguration(
        project_dir=checkpoint_dir,
        logging_dir=args.results_dir,
        automatic_checkpoint_naming=True,
        total_limit=args.ckpt_total_limit,
    )
    accelerator = Accelerator(project_config=project_config)
    device = accelerator.device

    # Create model:
    assert (
        args.image_size % 8 == 0
    ), "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    if "SEM" in args.model:
        model = DiT_models[args.model](
            input_size=latent_size, num_classes=args.num_classes, L=args.L, V=args.V
        )
    elif "CONT" in args.model:
        model = DiT_models[args.model](
            input_size=latent_size, num_classes=args.num_classes, in_dim=args.in_dim
        )
    else:
        model = DiT_models[args.model](
            input_size=latent_size, num_classes=args.num_classes
        )

    model = model.to(device)
    model = torch.compile(model)
    if args.pretrained_path is not None:
        if "safetensors" in args.pretrained_path:
            sd = load_file(args.pretrained_path)
        else:
            sd = torch.load(args.pretrained_path)
        msg = model.load_state_dict(sd)
        logger.info(
            f"Loading state from pretrained_path: {args.pretrained_path} with msg {msg}"
        )
        pass
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(
        device
    )  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    diffusion = create_diffusion(
        timestep_respacing=""
    )  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    if args.hf_dataset is not None:

        def process_fn(x):
            f, l = torch.tensor(x['features']), torch.tensor(x['labels'])
            return {"features": f, "labels": l}
        try:
            dataset = load_dataset(args.hf_dataset)
        except:
            dataset = load_from_disk(args.hf_dataset)
        # dataset = dataset.map(process_fn)
        logger.info(f"HF dataset: {args.hf_dataset}")
    else:
        features_dir = f"{args.feature_path}/imagenet256_features"
        labels_dir = f"{args.feature_path}/imagenet256_labels"
        sems_dir = f"{args.feature_path}/imagenet256_sem"
        logger.info(f"Feature directories: {features_dir}, {sems_dir}")
        if args.use_sem:
            dataset = CustomDataset(features_dir, sems_dir)
        else:
            dataset = CustomDataset(features_dir, labels_dir)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=True,
        collate_fn=collate_tensor_fn if args.hf_dataset is not None else None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images ({args.feature_path})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    model, opt, loader = accelerator.prepare(model, opt, loader)
    accelerator.register_for_checkpointing(ema)

    metadata_path = os.path.join(checkpoint_dir, "metadata.json")
    resume = os.path.exists(metadata_path)
    logger.info(f"Resuming checkpoint: {resume}")
    if resume:
        accelerator.load_state()
        with open(metadata_path, "r") as f:
            meta_data = json.load(f)
        current_epoch = meta_data["current_epoch"] + 1
        accelerator.project_configuration.iteration = current_epoch
    else:
        current_epoch = 0
        meta_data = {"current_epoch": current_epoch, "args": vars(args)}

    # accelerator.project_configuration.iteration = current_epoch + 1

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    if accelerator.is_main_process:
        logger.info(
            f"Training for {args.epochs} epochs... Starting from epoch {current_epoch}."
        )

    for epoch in range(current_epoch, args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        for x, y in tqdm(loader):
            x = x.to(device)
            y = y.to(device)
            x = x.squeeze(dim=1)
            # y = y.squeeze(dim=1)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item() / accelerator.num_processes
                if accelerator.is_main_process:
                    logger.info(
                        f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"
                    )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

        # Save DiT checkpoint:
        if epoch % args.ckpt_every == 0:
            accelerator.save_state()
            if accelerator.is_main_process:
                logger.info("Saving state and metadata")
                meta_data["current_epoch"] = epoch
                with open(metadata_path, "w") as f:
                    json.dump(meta_data, f)
            if args.exit_after_save:
                accelerator.wait_for_everyone()
                exit(0)

    accelerator.save_state()
    if accelerator.is_main_process:
        logger.info("Saving state and metadata")
        meta_data["current_epoch"] = epoch
        with open(metadata_path, "w") as f:
            json.dump(meta_data, f)

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    if accelerator.is_main_process:
        logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    args = parse_args()
    main(args)

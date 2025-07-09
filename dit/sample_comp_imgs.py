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
from models import DiT_models
from PIL import Image
from pipeline_dlc_dit import DLCDiTPipeline
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


class SubsetImageFolder(ImageFolder):
    def __init__(self, *args, classes, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples = [
            sample
            for sample, target in zip(self.samples, self.targets)
            if sample[0].split("/")[-1].split(".")[0] in classes
        ]
        self.targets = [
            target
            for sample, target in zip(self.samples, self.targets)
            if sample[0].split("/")[-1].split(".")[0] in classes
        ]


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


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    dino_sem = AutoModel.from_pretrained(
        "lavoies/SEM_dinov2_L512", trust_remote_code=True
    )
    dino_sem = dino_sem.eval()
    dino_sem = dino_sem.cuda()

    pipe = DLCDiTPipeline.from_pretrained(
        "lavoies/DLC_DiT_L512", trust_remote_code=True
    )
    pipe = pipe.to("cuda")

    # Labels to condition the model with (feel free to change):
    # class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    class_labels = args.class_id.split(",")
    # class_labels = list(map(int, class_labels))

    # Get all samples which has a class in class_labels
    dataset_path = args.imagenet_path
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = SubsetImageFolder(dataset_path, classes=class_labels, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=4, drop_last=False
    )
    sems = []
    dino_sem.temp = args.temp
    for sample, label in tqdm(loader):
        sem = dino_sem(sample.cuda()).sem
        sems.append(sem)
    sems = torch.cat(sems)
    sems = sems.mean(0).view(dino_sem.L, dino_sem.V)
    y = torch.stack(
        [torch.multinomial(p, num_samples=args.n, replacement=True) for p in sems], 1
    ).cuda()
    samples = pipe(dlcs=y, num_inference_steps=250).images
    # Encode the samples for each class with the SEM encoder
    # Get the SEM embeddings for each class
    # Generate samples for each class labels

    # idxs = [13000, 13001, 13002, 13003, 13004, 13005]

    # Create sampling noise:

    # Save and display images:
    save_path = os.path.join(args.save_path, f"{args.class_id}")
    os.makedirs(save_path, exist_ok=True)
    grid_save_path = os.path.join(save_path, f"{args.cfg_scale}-{args.temp}.png")
    # save_image(samples, grid_save_path, nrow=2, normalize=True, value_range=(-1, 1))
    for i, sample in enumerate(samples):
        img_save_path = os.path.join(save_path, f"{args.cfg_scale}-{args.temp}-{i}.png")
        sample.save(img_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2"
    )
    parser.add_argument("--class-id", type=str, default=207)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--save-path", type=str, default="figures")
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--temp", type=float, default=1)
    parser.add_argument(
        "--imagenet-path",
        type=str,
        default="/network/datasets/imagenet.var/imagenet_torchvision/train/",
    )
    args = parser.parse_args()
    main(args)

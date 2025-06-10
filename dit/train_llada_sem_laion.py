import os
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.multiprocessing as mp
import transformers
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from PIL import Image
from safetensors.torch import load_file
from tokenizers import AddedToken
from torchvision import transforms
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from transformers.loss.loss_utils import fixed_cross_entropy
from transformers.trainer_utils import get_last_checkpoint

from sft_llama.collator import DataCollatorForCompletionOnlyLM
from sft_llama.sem_llama import LlamaForCausalLMWithLearnedPositions
from sft_llama.sft_config import ModelConfig, ScriptArguments, SFTConfig
from sft_llama.sft_trainer_laion import SFTTrainer
from vision_transformer import vit_base_sem, vit_large_sem

transformers.logging.set_verbosity_debug()


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


@dataclass
class MyScriptArguments(ScriptArguments):
    soft_loss: bool = False
    reduce_lm_head: bool = False
    sem_embeddings: Literal["learned-pos", "separate-pos", "shared"] = "shared"
    L: int = 128
    V: int = 1024
    encoder_path: str = None
    model_name: str = "clm"
    # learned pos is for shared token for each SEM but learned position embed
    # separate is for separate tokens for each SEM at each position (e.g. SEM 1 at pos 1 != SEM 1 at pos 2)
    # shared is for shared tokens for each SEM, i.e. standard tokens


def get_peft_config(model_args: ModelConfig):
    if model_args.use_peft is False:
        return None

    peft_config = LoraConfig(
        task_type=model_args.lora_task_type,
        r=model_args.lora_r,
        target_modules=model_args.lora_target_modules,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        use_rslora=model_args.use_rslora,
        modules_to_save=model_args.lora_modules_to_save,
    )

    return peft_config


def tokenization(tokenizer):
    def fn(example):
        return tokenizer(example["json"]["caption"])

    return fn


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


class Collate:
    def __init__(self, formatting_func, tokenizer, model_type):
        self.formatting_func = formatting_func
        self.tokenizer = tokenizer
        self.model_type = model_type

    def __call__(self, examples):
        if "sem" in examples[0]:
            if self.model_type == "clm":
                texts_sem = self.formatting_func(examples)

                tokens = self.tokenizer(texts_sem, return_tensors="pt", padding=True)
                # The labels are the input_ids, and we mask the padding tokens in the loss computation
                labels = tokens["input_ids"].clone()
                labels[labels == self.tokenizer.pad_token_id] = -100  #
                # Ignore the image token index in the loss computation (model specific)
                image_token_id = self.tokenizer.convert_tokens_to_ids("<|start_sem|>")
                pos_start_sem = (labels == image_token_id).nonzero()
                for i, j in pos_start_sem:
                    labels[i, : j + 1] = -100
                tokens["labels"] = labels
                return tokens

            tokens = self.tokenizer(
                [example["text"][0] for example in examples],
                return_tensors="pt",
                padding=True,
            )
            labels = [example["sem_argmax"] for example in examples]
            labels = torch.tensor(labels)
            tokens["labels"] = labels
            return tokens

        else:
            return examples


def formatting_func(examples):
    output_string = []
    for example in examples:
        texts, sems_argmax = (
            example["text"],
            example["sem_argmax"],
        )
        first_text = texts[0]
        sem_token_strings = [f"<|sem_vocab_{i}|>" for i in sems_argmax]
        simple_sft_string = "".join(sem_token_strings) + "<|start_sem|>" + first_text

        output_string.append(simple_sft_string)
    return output_string


def convert_rgb(example):
    example["image"] = example["image"].convert("RGB")
    return example


class Transform:
    def __init__(self):
        self.transform_encoder = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, example):
        example["image"] = self.transform_encoder(example["image"])
        return example


def filter_no_caption_or_no_image(sample):
    has_caption = "txt" in sample
    has_image = (
        "png" in sample or "jpg" in sample or "jpeg" in sample or "webp" in sample
    )
    return has_caption and has_image


def main(script_args, training_args, model_args):
    ################
    # Dataset
    ################

    train_dataset = load_dataset(
        "webdataset",
        data_files={"train": "/network/datasets/laion400m/laion400m/*.tar"},
        split="train",
        streaming=True,
    )

    # train_dataset = train_dataset.map(preprocess_image)
    # train_dataset = train_dataset.filter(filter_no_caption_or_no_image)
    train_dataset = train_dataset.shuffle(seed=int(time.time()))
    train_dataset = train_dataset.rename_column("txt", "text")
    train_dataset = train_dataset.rename_column("jpg", "image")
    train_dataset = train_dataset.map(convert_rgb)
    train_dataset = train_dataset.map(Transform())

    # test_dataset = load_dataset(
    #     script_args.dataset_name, split=script_args.dataset_test_split
    # )
    L = script_args.L
    V = script_args.V

    ################
    # Model init kwargs & Tokenizer
    ################
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    if "mdlm" in model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            "gpt2",
            trust_remote_code=model_args.trust_remote_code,
            use_fast=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            use_fast=True,
        )

    if script_args.model_name == "clm":
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
        tokenizer.add_special_tokens({"sep_token": "<|start_sem|>"})
        tokenizer.add_tokens(
            [
                AddedToken(f"<|sem_vocab_{i}|>", single_word=True, normalized=False)
                for i in range(V)
            ]
        )
        compute_loss_func = None
        tokenizer.pad_token = tokenizer.eos_token
    elif script_args.model_name == "bert":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path, num_labels=L * V
        )

        def compute_loss_func(outputs, labels, num_items_in_batch=None):
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
            labels = labels.to(logits.device)

            # Flatten the tokens
            logits = logits.view(-1, V)
            labels = labels.view(-1)

            loss = fixed_cross_entropy(logits, labels, num_items_in_batch)
            return loss

    elif script_args.model_name == "diffusion":
        # model_kwargs["use_cache"] = False
        del model_kwargs["use_cache"]
        if "mdlm" in model_args.model_name_or_path:
            from mdlm.modeling_mdlm import MDLM

            model = MDLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
            tokenizer.mask_token_id = len(tokenizer)
            tokenizer.pad_token = tokenizer.eos_token
        else:
            model = AutoModel.from_pretrained(
                model_args.model_name_or_path, **model_kwargs
            )

        print("WARNING -- Currently hardcoding the mask index for LLaDA.")
        tokenizer.add_special_tokens({"sep_token": "<|start_sem|>"})
        tokenizer.add_tokens(
            [
                AddedToken(f"<|sem_vocab_{i}|>", single_word=True, normalized=False)
                for i in range(V)
            ]
        )
        model.config.tie_word_embeddings = model.config.weight_tying
        # model.model.set_activation_checkpointing("whole_layer")
        # model.forward = torch.compile(model.forward)

        def compute_loss_func(outputs, labels, p_mask, num_items_in_batch=None):
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
            labels = labels.to(logits.device)
            # loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')

            if num_items_in_batch is None:
                num_items_in_batch = labels.ne(-100).sum()
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none"
            )
            loss = loss  # / p_mask.view(-1)
            return loss.sum() / num_items_in_batch
    else:
        raise NotImplementedError("Not implemented")
    # torch._dynamo.config.optimize_ddp = False
    # model.forward = torch.compile(model.forward)

    dino_sem = vit_large_sem(
        img_size=518,
        patch_size=14,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        drop_path_rate=0.3,
        init_values=1.0e-5,
        L=script_args.L,
        V=script_args.V,
        drop_path_uniform=True,
    )
    dino_sem = dino_sem.eval()
    # dino_sem = dino_sem.cuda()
    dino_sem.sem_out = torch.nn.Identity()
    dino_sem.sem_cls_norm = torch.nn.Identity()
    dino_sem.norm = torch.nn.Identity()
    dino_sem = load_pretrained(dino_sem, script_args.encoder_path)
    for param in dino_sem.parameters():
        param.requires_grad = False
    dino_sem = torch.compile(dino_sem)

    data_collator = Collate(formatting_func, tokenizer, training_args.model_type)

    ################
    # Training
    ################
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    trainer = SFTTrainer(
        model=model,
        dino_sem=dino_sem,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        compute_loss_func=compute_loss_func,
        # peft_config=get_peft_config(model_args),
        # eval_dataset=test_dataset,
    )

    trainer.model.resize_token_embeddings(len(tokenizer))

    unwrap_model = trainer.accelerator.unwrap_model(trainer.model)
    if isinstance(unwrap_model, PeftModel):
        for param in trainer.accelerator.unwrap_model(
            trainer.model
        ).model.model.embed_tokens.parameters():
            param.requires_grad = True

    # Print trainable parameter count for verificatiofsdpn
    trainable_params = sum(
        p.numel() for p in trainer.model.parameters() if p.requires_grad
    )
    print(f"Number of trainable parameters: {trainable_params:,}")

    # if there's a last checkpoint, resume
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    # We have to do this hack because Trainer is a terrible abstraction.
    trainer._signature_columns = [
        "image",
        "text",
        "sem",
        "sem_argmax",
        "input_ids",
        "labels",
    ]
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Save and push to hub
    if training_args.save_strategy != "no":
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

        # Save merged
        unwrap_model = trainer.accelerator.unwrap_model(trainer.model)
        if isinstance(unwrap_model, PeftModel):
            model = unwrap_model.merge_and_unload()

            output_merged_dir = os.path.join(training_args.output_dir, "final_merged")
            model.save_pretrained(output_merged_dir, safe_serialization=True)
            tokenizer.save_pretrained(training_args.output_merged_dir)


if __name__ == "__main__":
    parser = HfArgumentParser((MyScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    main(script_args, training_args, model_args)

# Modified from TRL SFT Trainer
# Copyright 2025 The HuggingFace Team. All rights reserved.

import dataclasses
import inspect
import os
import warnings
from typing import Callable, Optional, Union

import datasets
import torch
import torch.nn as nn
from accelerate.state import PartialState
from datasets import Dataset
from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BaseImageProcessor,
    DataCollator,
    DataCollatorForLanguageModeling,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
)
from transformers.trainer_utils import EvalPrediction

from .sft_config import SFTConfig
from .trainer_callback import TrainerCallback


def peft_module_casting_to_bf16(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm) or "norm" in name:
            module = module.to(torch.float32)
        elif any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)


class SFTTrainer(Trainer):
    r"""
    Class definition of the Supervised Finetuning Trainer (SFT Trainer).
    This class is a wrapper around the `transformers.Trainer` class and inherits all of its attributes and methods.
    The trainer takes care of properly initializing the PeftModel in case a user passes a `PeftConfig` object.

    Args:
        model (Union[`transformers.PreTrainedModel`, `nn.Module`, `str`]):
            The model to train, can be a `PreTrainedModel`, a `torch.nn.Module` or a string with the model name to
            load from cache or download. The model can be also converted to a `PeftModel` if a `PeftConfig` object is
            passed to the `peft_config` argument.
        args (`Optional[SFTConfig]`):
            The arguments to tweak for training. Will default to a basic instance of [`SFTConfig`] with the `output_dir`
            set to a directory named *tmp_trainer* in the current directory if not provided.
        data_collator (`Optional[transformers.DataCollator]`):
            The data collator to use for training.
        train_dataset (`Optional[datasets.Dataset]`):
            The dataset to use for training. We recommend users to use `trl.trainer.ConstantLengthDataset` to create their dataset.
        eval_dataset (Optional[Union[`datasets.Dataset`, dict[`str`, `datasets.Dataset`]]]):
            The dataset to use for evaluation. We recommend users to use `trl.trainer.ConstantLengthDataset` to create their dataset.
        processing_class (`PreTrainedTokenizerBase` or `BaseImageProcessor` or `FeatureExtractionMixin` or `ProcessorMixin`, *optional*):
            Processing class used to process the data. If provided, will be used to automatically process the inputs
            for the model, and it will be saved along the model to make it easier to rerun an interrupted training or
            reuse the fine-tuned model.
            This supercedes the `tokenizer` argument, which is now deprecated.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        compute_metrics (`Callable[[transformers.EvalPrediction], dict]`, *optional* defaults to None):
            The function used to compute metrics during evaluation. It should return a dictionary mapping metric names to metric values.
            If not specified, only the loss will be computed during evaluation.
        callbacks (`list[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        formatting_func (`Optional[Callable]`):
            The formatting function to be used for creating the `ConstantLengthDataset`.
    """

    _tag_names = ["trl", "sft"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[SFTConfig] = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[
                PreTrainedTokenizerBase,
                BaseImageProcessor,
                FeatureExtractionMixin,
                ProcessorMixin,
            ]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        peft_config: Optional["PeftConfig"] = None,
        formatting_func: Optional[Callable] = None,
    ):
        if args is None:
            args = SFTConfig(output_dir="tmp_trainer")
        elif args is not None and args.__class__.__name__ == "TrainingArguments":
            args_as_dict = args.to_dict()
            # Manually copy token values as TrainingArguments.to_dict() redacts them
            args_as_dict.update(
                {
                    k: getattr(args, k)
                    for k in args_as_dict.keys()
                    if k.endswith("_token")
                }
            )
            args = SFTConfig(**args_as_dict)

        if getattr(args, "model_init_kwargs", None) is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            pass
        else:
            model_init_kwargs = args.model_init_kwargs
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if torch_dtype is not None:
                # Convert to `torch.dtype` if an str is passed
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)
                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the SFTConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )
                model_init_kwargs["torch_dtype"] = torch_dtype

            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

            if peft_config is not None:
                if not isinstance(peft_config, PeftConfig):
                    raise ValueError(
                        "If you want to use the PeftModel, you need to pass a PeftConfig object to the SFTTrainer."
                        f" and you passed a {type(peft_config)}."
                    )

                if not isinstance(model, PeftModel):
                    _support_gc_kwargs = hasattr(
                        args, "gradient_checkpointing_kwargs"
                    ) and "gradient_checkpointing_kwargs" in list(
                        inspect.signature(prepare_model_for_kbit_training).parameters
                    )
                    gradient_checkpointing_kwargs = (
                        getattr(args, "gradient_checkpointing_kwargs", None) or {}
                    )
                    is_sharded_qlora = False
                    # Below is to support QLoRA + FSDP / DS-Zero3 - one should never call
                    # peft_module_casting_to_bf16 or prepare_model_for_kbit_training when doing
                    # QLoRA + FSDP / DS-Zero3
                    if getattr(model, "is_loaded_in_4bit", False):
                        for _, param in model.named_parameters():
                            if param.__class__.__name__ == "Params4bit":
                                is_sharded_qlora = param.data.device.type in {
                                    "cpu",
                                    "meta",
                                }
                                break
                    if getattr(model, "is_loaded_in_8bit", False) or (
                        getattr(model, "is_loaded_in_4bit", False)
                        and not is_sharded_qlora
                    ):
                        prepare_model_kwargs = {
                            "use_gradient_checkpointing": getattr(
                                args, "gradient_checkpointing", False
                            )
                        }

                        if _support_gc_kwargs:
                            prepare_model_kwargs["gradient_checkpointing_kwargs"] = (
                                gradient_checkpointing_kwargs
                            )

                        model = prepare_model_for_kbit_training(
                            model, **prepare_model_kwargs
                        )

                        if args is not None:
                            args = dataclasses.replace(
                                args, gradient_checkpointing=False
                            )
                    elif getattr(args, "gradient_checkpointing", False) and (
                        "use_reentrant" not in gradient_checkpointing_kwargs
                        or gradient_checkpointing_kwargs["use_reentrant"]
                    ):
                        # For backward compatibility with older versions of transformers
                        if hasattr(model, "enable_input_require_grads"):
                            model.enable_input_require_grads()
                        else:

                            def make_inputs_require_grad(module, input, output):
                                output.requires_grad_(True)

                            model.get_input_embeddings().register_forward_hook(
                                make_inputs_require_grad
                            )

                    if (
                        "autocast_adapter_dtype"
                        in list(inspect.signature(get_peft_model).parameters)
                        and getattr(model, "is_loaded_in_4bit", False)
                        and is_sharded_qlora
                    ):
                        model = get_peft_model(
                            model, peft_config, autocast_adapter_dtype=False
                        )
                    else:
                        model = get_peft_model(model, peft_config)
                    if (
                        args is not None
                        and args.bf16
                        and getattr(model, "is_loaded_in_4bit", False)
                        and not is_sharded_qlora
                    ):
                        peft_module_casting_to_bf16(model)

        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path)
            if getattr(processing_class, "pad_token", None) is None:
                processing_class.pad_token = processing_class.eos_token

        if args.max_seq_length is None:
            # to overcome some issues with broken tokenizers
            args.max_seq_length = min(processing_class.model_max_length, 1024)

        self.dataset_num_proc = args.dataset_num_proc
        self.dataset_batch_size = args.dataset_batch_size

        if args.dataset_kwargs is None:
            args.dataset_kwargs = {}

        if not args.packing:
            if data_collator is None:
                data_collator = DataCollatorForLanguageModeling(
                    tokenizer=processing_class, mlm=False
                )

        # Pre-process the datasets only once per node. The remaining processes will use the cache.
        with PartialState().local_main_process_first():
            if train_dataset is not None:
                train_dataset = self._prepare_dataset(
                    train_dataset,
                    processing_class,
                    args.packing,
                    args.dataset_text_field,
                    args.max_seq_length,
                    formatting_func,
                    args.num_of_sequences,
                    args.chars_per_token,
                    remove_unused_columns=args.remove_unused_columns
                    if args is not None
                    else True,
                    **args.dataset_kwargs,
                )
            if eval_dataset is not None:
                _multiple = isinstance(eval_dataset, dict)
                _eval_datasets = (
                    eval_dataset if _multiple else {"singleton": eval_dataset}
                )

                eval_packing = (
                    args.packing if args.eval_packing is None else args.eval_packing
                )

                for _eval_dataset_name, _eval_dataset in _eval_datasets.items():
                    _eval_datasets[_eval_dataset_name] = self._prepare_dataset(
                        _eval_dataset,
                        processing_class,
                        eval_packing,
                        args.dataset_text_field,
                        args.max_seq_length,
                        formatting_func,
                        args.num_of_sequences,
                        args.chars_per_token,
                        remove_unused_columns=args.remove_unused_columns
                        if args is not None
                        else True,
                        **args.dataset_kwargs,
                    )
                if not _multiple:
                    eval_dataset = _eval_datasets["singleton"]

        if (
            processing_class.padding_side is not None
            and processing_class.padding_side != "right"
        ):
            warnings.warn(
                "You passed a processing_class with `padding_side` not equal to `right` to the SFTTrainer. This might "
                "lead to some unexpected behaviour due to overflow issues when training a model in half-precision. "
                "You might consider adding `processing_class.padding_side = 'right'` to your code.",
                UserWarning,
            )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if self.train_dataset is not None:
            if self.args.max_steps > 0 and args.packing:
                self.train_dataset.infinite = True
            elif self.args.max_steps == -1 and args.packing:
                self.train_dataset.infinite = False

    def _prepare_dataset(
        self,
        dataset,
        processing_class,
        packing,
        dataset_text_field: str,
        max_seq_length,
        formatting_func: Optional[Callable],
        num_of_sequences,
        chars_per_token,
        remove_unused_columns=True,
        append_concat_token=True,
        add_special_tokens=True,
        skip_prepare_dataset=False,
    ):
        if dataset is None:
            raise ValueError("The dataset should not be None")

        if skip_prepare_dataset:
            return dataset

        # If the dataset is already preprocessed (tokenized), return as-is. Only works if dataset is
        # a datasets.Dataset or datasets.IterableDataset -- not for torch Dataset
        column_names = (
            dataset.column_names
            if isinstance(dataset, (datasets.Dataset, datasets.IterableDataset))
            else None
        )
        if column_names and "input_ids" in column_names:
            if formatting_func is not None:
                warnings.warn(
                    "You passed a dataset that is already processed (contains an `input_ids` field) together with a "
                    "valid formatting function. Therefore `formatting_func` will be ignored. Either remove the "
                    "`formatting_func` or pass a dataset that is not already processed.",
                    UserWarning,
                )

            def formatting_func(x):
                return x["input_ids"]

            if not packing:
                return dataset

        # check if torch dataset / dataloader and do nothing
        # see https://github.com/huggingface/trl/pull/1468 for why datasets.IterableDataset needs a separate check
        if isinstance(
            dataset,
            (
                torch.utils.data.IterableDataset,
                torch.utils.data.Dataset,
                # ConstantLengthDataset,
            ),
        ) and not isinstance(dataset, datasets.IterableDataset):
            return dataset

        if not packing:
            return self._prepare_non_packed_dataloader(
                processing_class,
                dataset,
                dataset_text_field,
                max_seq_length,
                formatting_func,
                add_special_tokens,
                remove_unused_columns,
            )

        else:
            raise NotImplementedError

    def _prepare_non_packed_dataloader(
        self,
        processing_class,
        dataset,
        dataset_text_field: str,
        max_seq_length,
        formatting_func: Optional[Callable] = None,
        add_special_tokens=True,
        remove_unused_columns=True,
    ):
        # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
        def tokenize(element):
            outputs = processing_class(
                element[dataset_text_field]
                if formatting_func is None
                else formatting_func(element),
                add_special_tokens=add_special_tokens,
                truncation=True,
                padding=False,
                max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,
            )

            if formatting_func is not None and not isinstance(
                formatting_func(element), list
            ):
                raise ValueError(
                    "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                )

            return {
                "input_ids": outputs["input_ids"],
                "attention_mask": outputs["attention_mask"],
            }

        signature_columns = ["input_ids", "labels", "attention_mask"]

        if dataset.column_names is not None:  # None for IterableDataset
            extra_columns = list(set(dataset.column_names) - set(signature_columns))
        else:
            extra_columns = []

        if not remove_unused_columns and len(extra_columns) > 0:
            warnings.warn(
                "You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with "
                "the default collator and yield to errors. If you want to inspect dataset other columns (in this "
                f"case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you used the "
                "default collator and create your own data collator in order to inspect the unused dataset columns.",
                UserWarning,
            )

        map_kwargs = {
            "batched": True,
            "remove_columns": dataset.column_names if remove_unused_columns else None,
            "batch_size": self.dataset_batch_size,
        }
        if isinstance(dataset, datasets.Dataset):
            map_kwargs["num_proc"] = (
                self.dataset_num_proc
            )  # this arg is not available for IterableDataset
        tokenized_dataset = dataset.map(tokenize, **map_kwargs)

        return tokenized_dataset

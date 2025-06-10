from dataclasses import dataclass, field
from typing import Any, Optional

from transformers import TrainingArguments


@dataclass
class SFTConfig(TrainingArguments):
    r"""
    Configuration class for the [`SFTTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        dataset_text_field (`str`, *optional*, defaults to `"text"`):
            Name of the text field of the dataset. If provided, the trainer will automatically create a
            [`ConstantLengthDataset`] based on `dataset_text_field`.
        packing (`bool`, *optional*, defaults to `False`):
            Controls whether the [`ConstantLengthDataset`] packs the sequences of the dataset.
        learning_rate (`float`, *optional*, defaults to `2e-5`):
            Initial learning rate for [`AdamW`] optimizer. The default value replaces that of [`~transformers.TrainingArguments`].
        max_seq_length (`int` or `None`, *optional*, defaults to `None`):
            Maximum sequence length for the [`ConstantLengthDataset`] and for automatically creating the dataset. If
            `None`, it uses the smaller value between `tokenizer.model_max_length` and `1024`.
        dataset_num_proc (`int` or `None`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset. Only used when `packing=False`.
        dataset_batch_size (`Union[int, None]`, *optional*, defaults to `1000`):
            Number of examples to tokenize per batch. If `dataset_batch_size <= 0` or `dataset_batch_size is None`,
            tokenizes the full dataset as a single batch.
        model_init_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the model from a
            string.
        dataset_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Dictionary of optional keyword arguments to pass when creating packed or non-packed datasets.
        eval_packing (`bool` or `None`, *optional*, defaults to `None`):
            Whether to pack the eval dataset. If `None`, uses the same value as `packing`.
        num_of_sequences (`int`, *optional*, defaults to `1024`):
            Number of sequences to use for the [`ConstantLengthDataset`].
        chars_per_token (`float`, *optional*, defaults to `3.6`):
            Number of characters per token to use for the [`ConstantLengthDataset`]. See
            [chars_token_ratio](https://github.com/huggingface/trl/blob/08f550674c553c36c51d1027613c29f14f3676a5/examples/stack_llama/scripts/supervised_finetuning.py#L53) for more details.
        use_liger (`bool`, *optional*, defaults to `False`):
            Monkey patch the model with Liger kernels to increase throughput and reduce memory usage.
    """

    dataset_text_field: str = field(
        default="text",
        metadata={
            "help": "Name of the text field of the dataset. If provided, the trainer will automatically create a "
            "`ConstantLengthDataset` based on `dataset_text_field`."
        },
    )
    model_type: str = field(
        default="clm",
        metadata={"help": "Control whether the model is a CLM or a BERT model."},
    )
    packing: bool = field(
        default=False,
        metadata={
            "help": "Controls whether the `ConstantLengthDataset` packs the sequences of the dataset."
        },
    )
    learning_rate: float = field(
        default=2.0e-5,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`TrainingArguments`."
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum sequence length for the `ConstantLengthDataset` and for automatically creating the "
            "dataset. If `None`, it uses the smaller value between `tokenizer.model_max_length` and `1024`."
        },
    )
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of processes to use for processing the dataset. Only used when `packing=False`."
        },
    )
    dataset_batch_size: int = field(
        default=1000,
        metadata={
            "help": "Number of examples to tokenize per batch. If `dataset_batch_size <= 0` or `dataset_batch_size is "
            "None`, tokenizes the full dataset as a single batch."
        },
    )
    model_init_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the model "
            "from a string."
        },
    )
    dataset_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Dictionary of optional keyword arguments to pass when creating packed or non-packed datasets."
        },
    )
    eval_packing: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to pack the eval dataset. If `None`, uses the same value as `packing`."
        },
    )
    num_of_sequences: int = field(
        default=1024,
        metadata={
            "help": "Number of sequences to use for the `ConstantLengthDataset`."
        },
    )
    chars_per_token: float = field(
        default=3.6,
        metadata={
            "help": "Number of characters per token to use for the `ConstantLengthDataset`."
        },
    )
    use_liger: bool = field(
        default=False,
        metadata={
            "help": "Monkey patch the model with Liger kernels to increase throughput and reduce memory usage."
        },
    )
    use_stateful_dataloader: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to have the dataloaders prepared by the Accelerator be backed by`[torchdata.StatefulDataLoader]`(https://github.com/pytorch/data/tree/main/torchdata/stateful_dataloader). This requires `accelerate` version 1.0.0 or higher, and `torchdata` version 0.8.0 to be installed."
        },
    )


@dataclass
class ModelConfig:
    """
    Configuration class for the models.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        model_name_or_path (`str` or `None`, *optional*, defaults to `None`):
            Model checkpoint for weights initialization.
        model_revision (`str`, *optional*, defaults to `"main"`):
            Specific model version to use. It can be a branch name, a tag name, or a commit id.
        torch_dtype (`Literal["auto", "bfloat16", "float16", "float32"]` or `None`, *optional*, defaults to `None`):
            Override the default `torch.dtype` and load the model under this dtype. Possible values are

                - `"bfloat16"`: `torch.bfloat16`
                - `"float16"`: `torch.float16`
                - `"float32"`: `torch.float32`
                - `"auto"`: Automatically derive the dtype from the model's weights.

        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether to allow for custom models defined on the Hub in their own modeling files. This option should only
            be set to `True` for repositories you trust and in which you have read the code, as it will execute code
            present on the Hub on your local machine.
        attn_implementation (`str` or `None`, *optional*, defaults to `None`):
            Which attention implementation to use. You can run `--attn_implementation=flash_attention_2`, in which case
            you must install this manually by running `pip install flash-attn --no-build-isolation`.
        use_peft (`bool`, *optional*, defaults to `False`):
            Whether to use PEFT for training.
        lora_r (`int`, *optional*, defaults to `16`):
            LoRA R value.
        lora_alpha (`int`, *optional*, defaults to `32`):
            LoRA alpha.
        lora_dropout (`float`, *optional*, defaults to `0.05`):
            LoRA dropout.
        lora_target_modules (`Union[str, list[str]]` or `None`, *optional*, defaults to `None`):
            LoRA target modules.
        lora_modules_to_save (`list[str]` or `None`, *optional*, defaults to `None`):
            Model layers to unfreeze & train.
        lora_task_type (`str`, *optional*, defaults to `"CAUSAL_LM"`):
            Task type to pass for LoRA (use `"SEQ_CLS"` for reward modeling).
        use_rslora (`bool`, *optional*, defaults to `False`):
            Whether to use Rank-Stabilized LoRA, which sets the adapter scaling factor to `lora_alpha/√r`, instead of
            the original default value of `lora_alpha/r`.
        load_in_8bit (`bool`, *optional*, defaults to `False`):
            Whether to use 8 bit precision for the base model. Works only with LoRA.
        load_in_4bit (`bool`, *optional*, defaults to `False`):
            Whether to use 4 bit precision for the base model. Works only with LoRA.
        bnb_4bit_quant_type (`str`, *optional*, defaults to `"nf4"`):
            Quantization type (`"fp4"` or `"nf4"`).
        use_bnb_nested_quant (`bool`, *optional*, defaults to `False`):
            Whether to use nested quantization.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Model checkpoint for weights initialization."},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "Specific model version to use. It can be a branch name, a tag name, or a commit id."
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override the default `torch.dtype` and load the model under this dtype.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        },
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which attention implementation to use. You can run `--attn_implementation=flash_attention_2`, in "
            "which case you must install this manually by running `pip install flash-attn --no-build-isolation`."
        },
    )
    use_peft: bool = field(
        default=False,
        metadata={"help": "Whether to use PEFT for training."},
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA R value."},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout."},
    )
    lora_target_modules: Optional[list[str]] = field(
        default=None,
        metadata={"help": "LoRA target modules."},
    )
    lora_modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Model layers to unfreeze & train."},
    )
    lora_task_type: str = field(
        default="CAUSAL_LM",
        metadata={
            "help": "Task type to pass for LoRA (use 'SEQ_CLS' for reward modeling)."
        },
    )
    use_rslora: bool = field(
        default=False,
        metadata={
            "help": "Whether to use Rank-Stabilized LoRA, which sets the adapter scaling factor to `lora_alpha/√r`, "
            "instead of the original default value of `lora_alpha/r`."
        },
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={
            "help": "Whether to use 8 bit precision for the base model. Works only with LoRA."
        },
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={
            "help": "Whether to use 4 bit precision for the base model. Works only with LoRA."
        },
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization type.", "choices": ["fp4", "nf4"]},
    )
    use_bnb_nested_quant: bool = field(
        default=False,
        metadata={"help": "Whether to use nested quantization."},
    )

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")

        if (
            hasattr(self.lora_target_modules, "__len__")
            and len(self.lora_target_modules) == 1
        ):
            self.lora_target_modules = self.lora_target_modules[0]


@dataclass
class ScriptArguments:
    """
    Arguments common to all scripts.

    Args:
        dataset_name (`str`):
            Dataset name.
        dataset_config (`str` or `None`, *optional*, defaults to `None`):
            Dataset configuration name. Corresponds to the `name` argument of the [`~datasets.load_dataset`] function.
        dataset_train_split (`str`, *optional*, defaults to `"train"`):
            Dataset split to use for training.
        dataset_test_split (`str`, *optional*, defaults to `"test"`):
            Dataset split to use for evaluation.
        gradient_checkpointing_use_reentrant (`bool`, *optional*, defaults to `False`):
            Whether to apply `use_reentrant` for gradient checkpointing.
        ignore_bias_buffers (`bool`, *optional*, defaults to `False`):
            Debug argument for distributed training. Fix for DDP issues with LM bias/mask buffers - invalid scalar
            type, inplace operation. See https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992.
    """

    dataset_name: str = field(metadata={"help": "Dataset name."})
    dataset_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset configuration name. Corresponds to the `name` argument of the `datasets.load_dataset` "
            "function."
        },
    )
    dataset_train_split: str = field(
        default="train", metadata={"help": "Dataset split to use for training."}
    )
    dataset_test_split: str = field(
        default="test", metadata={"help": "Dataset split to use for evaluation."}
    )

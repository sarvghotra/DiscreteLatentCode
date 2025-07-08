from dataclasses import asdict, dataclass

import torch
from safetensors import safe_open
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    HfArgumentParser,
    LlamaConfig,
    LogitsProcessor,
    LogitsProcessorList,
    ModernBertConfig,
)

from generate_llada import generate_image


@dataclass
class ScriptArguments:
    model_name_or_path: str = None
    output_path: str = None
    batch_size: int = 1
    restrict_to_sems: bool = False
    prompt: str = "a zebra in the snow"
    num_samples: int = 6
    L: int = 4096
    V: int = 32
    # dummy variable
    cfg_scale: float = None


@dataclass
class GenerationArguments:
    do_sample: bool = True
    num_beams: int = 1
    num_beam_groups: int = 1
    temperature: float = 1.0
    steps: int = 512
    remasking: str = "low_confidence"  # random or low_confidence
    top_k: int = 50
    top_p: float = 1.0
    num_return_sequences: int = 6


class TokenRangeRestrictor(LogitsProcessor):
    """
    Custom LogitsProcessor that restricts token generation to a specific range.
    Sets logits for all tokens outside the specified range to negative infinity.
    """

    def __init__(self, min_id: int = 0, max_id: int = 100):
        self.min_id = min_id
        self.max_id = max_id

    def __call__(self, input_ids, scores):
        # Set logits outside our desired range to -inf
        scores[:, : self.min_id] = float("-inf")
        scores[:, self.max_id + 1 :] = float("-inf")
        return scores


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, GenerationArguments))
    args, generation_args = parser.parse_args_into_dataclasses()

    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    print(f"Prompt={args.prompt}")
    text_input = [args.prompt]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    sem_start_id = tokenizer.convert_tokens_to_ids("<|start_sem|>")
    if sem_start_id is None:
        tokenizer.add_special_tokens({"sep_token": "<|start_sem|>"})
        sem_start_id = tokenizer.convert_tokens_to_ids("<|start_sem|>")
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
    )
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        embedding_size=config.vocab_size,
        **model_kwargs,
    )
    generate_kwargs = {
        "steps": generation_args.steps,
        "gen_length": args.L,
        "block_length": args.L,
        "temperature": generation_args.temperature,
        "cfg_scale": 0,
        # "remasking": "low_confidence",
        "remasking": generation_args.remasking,
    }
    model.to("cuda")
    model.eval()
    completion_tensors = []
    for text in tqdm(text_input * args.num_samples):
        # model.generation_pos = 1
        inputs = tokenizer(["<|start_sem|>" + text], return_tensors="pt").to("cuda")
        outputs = generate_image(model, inputs.input_ids, **generate_kwargs)
        completions = [output[: args.L].cpu() for output in outputs]
        completion_tensors.extend(completions)

    completion_tensors = torch.stack(completion_tensors)
    sem_values = completion_tensors - sem_start_id - 1

    torch.save(sem_values, args.output_path)

    print("done")

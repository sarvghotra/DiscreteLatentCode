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
from sft_llama.sem_llama import LlamaForCausalLMWithLearnedPositions


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

    text_input = [args.prompt]

    if isinstance(config, BertConfig) or isinstance(config, ModernBertConfig):
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path
        )
        model.to("cuda")
        sem_values = []
        for text in tqdm(text_input):
            inputs = tokenizer(text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                output = model(**inputs)

            sem_logits = (
                output.logits.view(-1, args.L, args.V)[0] / generation_args.temperature
            )
            sems = torch.softmax(sem_logits, -1)
            sem_argmax = [
                torch.multinomial(
                    sem, num_samples=args.num_samples, replacement=True
                ).squeeze()
                for sem in sems
            ]
            sem_values.append(torch.stack(sem_argmax, 1))

        sem_values = torch.stack(sem_values).squeeze(0)

    elif isinstance(config, LlamaConfig) and config.architectures == [
        "LlamaForCausalLMWithLearnedPositions"
    ]:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        model = LlamaForCausalLMWithLearnedPositions.from_pretrained(
            args.model_name_or_path, **model_kwargs
        )
        sem_start_id = tokenizer.convert_tokens_to_ids("<|start_sem|>")
        model.start_token_id = sem_start_id

        model.to("cuda")

        # pipe = pipeline(
        #     "text-generation", model=args.model_name_or_path, device=0, batch_size=2
        # )

        generate_kwargs = {
            "max_new_tokens": args.L,
            "do_sample": True,
            "temperature": generation_args.temperature,
        }
        if args.restrict_to_sems:
            sep_sem_id = tokenizer.convert_tokens_to_ids("<|start_sem|>")
            model.start_token_id = sep_sem_id
            last_sem_id = sem_start_id + 32
            restrict_to_sems = TokenRangeRestrictor(
                min_id=sem_start_id, max_id=last_sem_id
            )
            logits_processor = LogitsProcessorList([restrict_to_sems])
            generate_kwargs["logits_processor"] = logits_processor
        # maybe also renormalize_logits=True

        # text_input = input("Enter text:")
        # text_input = ["An apple to the left of a pear" for _ in range(6)]
        completion_tensors = []
        # for outputs in tqdm(
        #     pipe(
        #         [text + "<|start_sem|>" for text in text_input],
        #         return_tensors=True,
        #         **generate_kwargs,
        #         batch_size=2,
        #     )
        # ):
        model.set_generation_mode()
        for text in tqdm(text_input):
            model.generation_pos = 1
            inputs = tokenizer([text + "<|start_sem|>"], return_tensors="pt").to("cuda")
            output = model.generate_image(**inputs, **generate_kwargs)
            completion = output[0][inputs["input_ids"].shape[1] :]
            completion_tensors.append(completion.cpu())
            # for out in outputs:
            # token_list = out["generated_token_ids"]
            # start_idx = token_list.index(sep_sem_id)
            # completion_tensors.append(token_list[start_idx + 1 :])

        completion_tensors = torch.stack(completion_tensors)
        sem_values = completion_tensors - sem_start_id - 1
    elif isinstance(config, LlamaConfig) and config.architectures == [
        "LlamaForCausalLM"
    ]:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        sem_start_id = tokenizer.convert_tokens_to_ids("<|start_sem|>")
        if sem_start_id is None:
            tokenizer.add_special_tokens({"sep_token": "<|start_sem|>"})
            sem_start_id = tokenizer.convert_tokens_to_ids("<|start_sem|>")
        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            **model_kwargs,
            ignore_mismatched_sizes=True,
        )

        generate_kwargs = {
            "max_new_tokens": args.L,
            **asdict(generation_args),
        }
        model.to("cuda")
        model.eval()
        completion_tensors = []
        for text in tqdm(text_input):
            # model.generation_pos = 1
            inputs = tokenizer([text + "<|start_sem|>"], return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, **generate_kwargs, eos_token_id=None)
            completions = [
                output[inputs["input_ids"].shape[1] :].cpu() for output in outputs
            ]
            completion_tensors.extend(completions)

        completion_tensors = torch.stack(completion_tensors)
        sem_values = completion_tensors - sem_start_id - 1
    elif config.__class__.__name__ == "LLaDAConfig":
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
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
            "steps": 1024,
            "gen_length": 128,
            "block_length": 128,
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
    else:
        raise Exception("bad config?")

    torch.save(sem_values, args.output_path)

    print("done")

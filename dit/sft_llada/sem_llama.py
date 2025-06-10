import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaConfig
from typing import Optional, Tuple, Union, List
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import DynamicCache, Cache


class LlamaSEMConfig(LlamaConfig):
    def __init__(
        self, num_learned_embeddings: int = 5000, start_token_id: int = None, **kwargs
    ):
        self.num_learned_embeddings = num_learned_embeddings
        self.start_token_id = start_token_id

        super().__init__(**kwargs)


class LlamaForCausalLMWithLearnedPositions(LlamaForCausalLM):
    config_class = LlamaSEMConfig

    def __init__(self, config):
        super().__init__(config)
        # Create learned position embeddings
        self.learned_pos_embedding = nn.Embedding(
            config.num_learned_embeddings,
            config.hidden_size,
            padding_idx=0,
        )

        # Initialize with small values
        nn.init.normal_(self.learned_pos_embedding.weight, mean=0.0, std=0.02)

        self.start_token_id = config.start_token_id
        self.generation_mode = False
        self.generation_pos = None

    def set_generation_mode(self):
        self.generation_mode = True
        self.generation_pos = 1

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # Get base model outputs

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not self.generation_mode or self.generation_pos == 1:
            repeat_position_ids = position_ids.repeat(input_ids.size(0), 1)

            sem_position_starts = (
                (input_ids == self.start_token_id).nonzero()[:, 1].unsqueeze(1)
            )
            sem_position_ids = repeat_position_ids - sem_position_starts
            # index 0 should be exactly on the SEM start token
            # since we have 0 as our padding idx that is fine

            sem_position_ids[sem_position_ids < 0] = 0
            if input_ids.shape == attention_mask.shape:
                sem_position_ids[attention_mask == 0] = 0
        else:
            sem_position_ids = torch.full_like(
                input_ids, fill_value=self.generation_pos
            )

        sem_position_embeds = self.learned_pos_embedding(sem_position_ids)

        inputs_embeds = inputs_embeds + sem_position_embeds

        if self.generation_mode:
            self.generation_pos += 1

        return super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )

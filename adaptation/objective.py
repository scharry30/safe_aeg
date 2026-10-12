"""Loss on learner responses without generation-time safety guidance."""

import torch

from .tokens import native_response_eos, native_response_example


class ResponseObjective:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.eos = native_response_eos(
            tokenizer,
            model.generation_config.eos_token_id,
            model.config.vocab_size,
        )
        self.gradient_response_tokens = 0

    def __call__(self, row):
        example = native_response_example(
            self.tokenizer,
            row["prompt"],
            row["response"],
            self.eos,
            self.model.config.max_position_embeddings,
        )
        device = next(self.model.parameters()).device
        inputs = torch.tensor([example["input_ids"]], device=device)
        labels = torch.tensor([example["labels"]], device=device)
        loss = self.model(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            labels=labels,
            use_cache=False,
        ).loss
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite response loss")
        self.gradient_response_tokens += example["gradient_response_tokens"]
        return loss

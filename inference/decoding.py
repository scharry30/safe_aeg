"""Greedy generation with explicit termination and no prompt cropping."""

import torch


def cut_generated_tokens(tokens, eos_ids, pad_id):
    result = []
    terminal = False
    for token in tokens:
        if token in eos_ids:
            terminal = True
            break
        result.append(token)
    if not terminal and pad_id is not None:
        while result and result[-1] == pad_id:
            result.pop()
    return result, terminal


def generate_greedy(model, tokenizer, prompts, maximum, batch_size=8, device=None):
    if maximum <= 0 or batch_size <= 0:
        raise ValueError("Positive decoding budgets required")
    device = device if device is not None else next(model.parameters()).device
    previous_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.eval()
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos)
    results = []
    try:
        for offset in range(0, len(prompts), batch_size):
            batch = prompts[offset : offset + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=False,
                add_special_tokens=False,
                return_tensors="pt",
            )
            width = inputs["input_ids"].shape[1]
            if width + maximum > model.config.max_position_embeddings:
                raise ValueError("Complete input exceeds the model context")
            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
                if key in {"input_ids", "attention_mask"}
            }
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=maximum,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            if output.shape[0] != len(batch):
                raise ValueError("Generation batch size differs from assigned inputs")
            for index, tokens in enumerate(output[:, width:].tolist()):
                ids, ended = cut_generated_tokens(
                    tokens, eos_ids, tokenizer.pad_token_id
                )
                results.append(
                    dict(
                        text=tokenizer.decode(ids, skip_special_tokens=True),
                        input_tokens=int(inputs["attention_mask"][index].sum()),
                        output_tokens=len(ids),
                        output_budget_exhausted=not ended,
                    )
                )
    finally:
        tokenizer.padding_side = previous_padding
    return results

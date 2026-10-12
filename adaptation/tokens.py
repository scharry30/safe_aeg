"""Response masking and model-native training termination."""


def native_response_eos(tokenizer, generation_eos, vocab_size):
    ids = [generation_eos] if isinstance(generation_eos, int) else generation_eos
    if (
        not isinstance(ids, (list, tuple))
        or not ids
        or any(isinstance(i, bool) or not isinstance(i, int) for i in ids)
    ):
        raise ValueError("Explicit integer generation EOS identifiers required")
    if (
        isinstance(vocab_size, bool)
        or not isinstance(vocab_size, int)
        or vocab_size <= 0
    ):
        raise ValueError("Positive model vocabulary size required")
    if any(i < 0 or i >= vocab_size for i in ids):
        raise ValueError("Generation EOS lies outside the model vocabulary")
    eos = tokenizer.eos_token_id
    if isinstance(eos, bool) or not isinstance(eos, int) or eos not in ids:
        raise ValueError("Tokenizer EOS must match a model generation terminator")
    return eos


def native_response_example(tokenizer, prompt, response, eos, context_limit):
    if (
        not isinstance(prompt, str)
        or not isinstance(response, str)
        or not response.strip()
    ):
        raise ValueError("A textual prompt and nonempty owned response are required")
    if isinstance(eos, bool) or not isinstance(eos, int) or eos < 0:
        raise ValueError("Valid native response EOS required")
    prefix = tokenizer.apply_chat_template(
        [dict(role="user", content=prompt)],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(prefix, add_special_tokens=False)
    answer_ids = tokenizer.encode(response, add_special_tokens=False)
    if not answer_ids:
        raise ValueError("Owned response must contain at least one token")
    ids = prompt_ids + answer_ids + [eos]
    if len(ids) > context_limit:
        raise ValueError("No training input or target truncation")
    return dict(
        input_ids=ids,
        labels=[-100] * len(prompt_ids) + answer_ids + [eos],
        gradient_response_tokens=len(answer_ids) + 1,
    )

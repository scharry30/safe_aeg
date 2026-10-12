"""Immutable adapter checkpoints with recursive-state metadata."""

import random
from pathlib import Path

import torch
from adaptation.parameters import adapter_state, parameter_fingerprint, restore_adapter
from data.sources import file_sha256


def save_checkpoint(path, model, state, rng):
    adapter = adapter_state(model)
    payload = dict(
        adapter=adapter,
        adapter_hash=parameter_fingerprint(adapter),
        procedure=dict(state.procedure),
        feedback=dict(state.feedback),
        replay=list(state.replay.rows),
        replay_capacity=state.replay.capacity,
        history=list(state.history),
        rng_state=rng.getstate(),
    )
    # Exclusive creation preserves every intermediate generation checkpoint.
    with Path(path).open("xb") as stream:
        torch.save(payload, stream)
    return dict(
        adapter_hash=payload["adapter_hash"], checkpoint_sha256=file_sha256(path)
    )


def load_checkpoint(path, expected_sha256, model):
    from data.replay import ReplayBuffer

    from .state import EvolutionState

    if file_sha256(path) != expected_sha256:
        raise ValueError("Checkpoint content differs from its expected hash")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if parameter_fingerprint(saved["adapter"]) != saved["adapter_hash"]:
        raise ValueError("Checkpoint adapter fingerprint differs")
    restore_adapter(model, saved["adapter"])
    state = EvolutionState(
        procedure=saved["procedure"],
        feedback=saved["feedback"],
        replay=ReplayBuffer(saved["replay"], saved["replay_capacity"]),
        history=saved["history"],
    )
    rng = random.Random()
    rng.setstate(saved["rng_state"])
    return state, rng

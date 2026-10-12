"""Adapter snapshots, restoration, and parameter-state identities."""

import hashlib
import math
import random

import torch


def seed_parameters(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def adapter_state(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    }


def parameter_fingerprint(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        cpu = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(cpu.dtype).encode())
        digest.update(str(tuple(cpu.shape)).encode())
        digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def parameter_displacement(before, after):
    if set(before) != set(after):
        raise ValueError("Adapter parameter identities differ")
    return math.sqrt(
        sum(
            float((before[name].float() - after[name].float()).square().sum())
            for name in before
        )
    )


def restore_adapter(model, state):
    parameters = {
        name: value for name, value in model.named_parameters() if "lora_" in name
    }
    if not state or set(parameters) != set(state):
        raise ValueError("Adapter parameter identities differ")
    if any(parameters[name].shape != value.shape for name, value in state.items()):
        raise ValueError("Adapter parameter shapes differ")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state[name].to(parameter.device, dtype=parameter.dtype))
    if parameter_fingerprint(adapter_state(model)) != parameter_fingerprint(state):
        raise ValueError("Adapter restoration changed the parameter state")

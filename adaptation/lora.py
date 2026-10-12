"""Native adapters on attention query and value projections."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class NativeLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        if rank <= 0 or not 0 < alpha < float("inf"):
            raise ValueError("Positive adapter rank and scale required")
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.lora_A = nn.Parameter(
            torch.empty(
                rank,
                base.in_features,
                dtype=base.weight.dtype,
                device=base.weight.device,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                base.out_features,
                rank,
                dtype=base.weight.dtype,
                device=base.weight.device,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return (
            self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        )


def inject_lora(model, rank=16, alpha=32.0):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if name.rsplit(".", 1)[-1] in ("q_proj", "v_proj")
        and isinstance(module, nn.Linear)
    ]
    if not targets:
        raise ValueError("Model has no q_proj/v_proj linear modules")
    replaced = []
    for name, module in targets:
        prefix, _, attribute = name.rpartition(".")
        parent = model.get_submodule(prefix) if prefix else model
        setattr(parent, attribute, NativeLoRALinear(module, rank, alpha))
        replaced.append(name)
    return replaced

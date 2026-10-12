"""Behavior learning and credit-weighted writer consolidation."""

import gc
import math

import torch

from .parameters import adapter_state, parameter_fingerprint


class ResponseOptimizer:
    def __init__(self, model, objective, rng, schedule):
        self.model = model
        self.objective = objective
        self.rng = rng
        self.schedule = schedule
        self.credited_writer_steps = 0

    def optimize(
        self, corpus, steps, learning_rate, focus=None, writer=None, credit=0.0
    ):
        if not corpus or steps <= 0:
            raise ValueError("A nonempty corpus and positive step budget are required")
        if not math.isfinite(credit) or not 0 <= credit <= 1:
            raise ValueError("Writer credit must lie in [0, 1]")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("A finite positive learning rate is required")
        before = parameter_fingerprint(adapter_state(self.model))
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError("No trainable adapter parameters")
        self.model.train()
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0)
        weights = [
            self.schedule.focus_weight if focus and row["category"] == focus else 1
            for row in corpus
        ]
        losses = []
        writer_steps = 0
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            samples = self.rng.choices(
                corpus, weights=weights, k=self.schedule.accumulation
            )
            for row in samples:
                value = self.objective(row) / self.schedule.accumulation
                value.backward()
                total += float(value.detach())
            if writer is not None and credit > 0:
                value = self.objective(writer) * min(1.0, credit)
                value.backward()
                total += float(value.detach())
                writer_steps += 1
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, self.schedule.max_grad_norm
            )
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite gradient norm")
            optimizer.step()
            losses.append(total)
        self.model.eval()
        after = parameter_fingerprint(adapter_state(self.model))
        if before == after:
            raise ValueError("No actual parameter update")
        self.credited_writer_steps += writer_steps
        del optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return dict(
            steps=steps,
            learning_rate=learning_rate,
            writer_credit=credit,
            credited_writer_steps=writer_steps,
            before_adapter_hash=before,
            after_adapter_hash=after,
            mean_loss=sum(losses) / len(losses),
        )

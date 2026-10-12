"""Shared optimizer, collection, and decoding budgets."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingSchedule:
    generations: int = 3
    behavior_steps: int = 64
    consolidation_steps: int = 4
    learning_rate: float = 2e-5
    replay_capacity: int = 240
    accumulation: int = 4
    focus_weight: int = 4
    max_grad_norm: float = 1.0
    collection_attempts: int = 3
    minimum_targets_per_kind: int = 8
    response_budget: int = 512
    writer_budget: int = 256
    judge_budget: int = 384
    learner_batch_size: int = 8
    judge_batch_size: int = 8
    lora_rank: int = 16
    lora_alpha: float = 32.0
    seed: int = 11

    def __post_init__(self):
        budgets = (
            self.generations,
            self.behavior_steps,
            self.consolidation_steps,
            self.replay_capacity,
            self.accumulation,
            self.focus_weight,
            self.collection_attempts,
            self.minimum_targets_per_kind,
            self.response_budget,
            self.writer_budget,
            self.judge_budget,
            self.learner_batch_size,
            self.judge_batch_size,
            self.lora_rank,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in budgets
        ):
            raise ValueError("Positive integer training budgets required")
        if not all(
            0 < value < float("inf")
            for value in (self.learning_rate, self.max_grad_norm, self.lora_alpha)
        ):
            raise ValueError("Finite positive optimization parameters required")

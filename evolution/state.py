"""Procedure/failure inheritance with unconditional generation retention."""

from copy import deepcopy
from dataclasses import dataclass, field

from data.replay import ReplayBuffer


@dataclass
class EvolutionState:
    procedure: dict
    replay: ReplayBuffer
    feedback: dict = field(default_factory=dict)
    history: list = field(default_factory=list)

    def reset_context(self, fixed_procedure):
        # This intervention does not reset the learned model parameters.
        self.procedure = dict(fixed_procedure)
        self.feedback = {}

    def retain(self, record, training_feedback):
        if record["generation"] != len(self.history) + 1:
            raise ValueError("Generation records must remain consecutive")
        self.history.append(deepcopy(record))
        self.feedback = deepcopy(training_feedback)

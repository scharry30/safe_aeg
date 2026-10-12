"""Baselines and mechanism interventions under a shared schedule."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Method:
    name: str
    adaptive_procedure: bool = False
    writer_supervision: bool = False
    inherit_procedure: bool = True
    validity_only_credit: bool = False
    reuse_bootstrap: bool = False
    learning_rate: float | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("A method name is required")
        if self.writer_supervision and not self.adaptive_procedure:
            raise ValueError("Writer supervision requires procedure authoring")
        if self.validity_only_credit and not self.writer_supervision:
            raise ValueError("Validity-only credit requires writer supervision")
        if self.reuse_bootstrap and self.adaptive_procedure:
            raise ValueError("A static bootstrap control cannot adapt its procedure")
        if self.learning_rate is not None and (
            not math.isfinite(self.learning_rate) or self.learning_rate <= 0
        ):
            raise ValueError("A finite positive learning rate is required")


FULL = Method("safe_aeg", adaptive_procedure=True, writer_supervision=True)
VERIFIED_SELF_TRAINING = Method("verified_self_training")
STATIC_SAFETY_SFT = Method("static_safety_sft", reuse_bootstrap=True)
NO_WRITER = Method("without_writer_loss", adaptive_procedure=True)
NO_INHERITANCE = Method(
    "without_inherited_context",
    adaptive_procedure=True,
    writer_supervision=True,
    inherit_procedure=False,
)
VALIDITY_ONLY = Method(
    "validity_only_credit",
    adaptive_procedure=True,
    writer_supervision=True,
    validity_only_credit=True,
)
LOWER_RATE = Method(
    "learning_rate_1e5",
    adaptive_procedure=True,
    writer_supervision=True,
    learning_rate=1e-5,
)
HIGHER_RATE = Method(
    "learning_rate_5e5",
    adaptive_procedure=True,
    writer_supervision=True,
    learning_rate=5e-5,
)

MAIN_METHODS = (FULL, VERIFIED_SELF_TRAINING, STATIC_SAFETY_SFT)
MECHANISM_METHODS = (FULL, NO_WRITER, NO_INHERITANCE, VALIDITY_ONLY)
RATE_METHODS = (LOWER_RATE, FULL, HIGHER_RATE)

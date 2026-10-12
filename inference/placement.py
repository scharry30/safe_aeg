"""Temporarily swap a frozen verifier into the learner's device."""

from contextlib import contextmanager

import torch


@contextmanager
def verifier_device(student, verifier, device, swap=False):
    if not swap:
        yield
        return
    student_device = next(student.parameters()).device
    verifier_origin = next(verifier.parameters()).device
    try:
        student.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        verifier.to(device)
        yield
    finally:
        verifier.to(verifier_origin)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        student.to(student_device)

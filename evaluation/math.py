"""Strict and relaxed numeric exact match for capability retention."""

import re
from decimal import Decimal, InvalidOperation

PROMPT_PREFIX = (
    "Solve the following math problem step by step. End your response "
    'with a final line of the form "#### <number>" containing only '
    "the numerical answer.\n\n"
)
NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
STRICT = re.compile(r"^[ \t]*####[ \t]*(" + NUMBER + r")\s*\Z", re.MULTILINE)
RELAXED = re.compile(NUMBER)


def number_value(token):
    if token is None:
        return None
    try:
        value = Decimal(token.replace(",", ""))
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def extract_strict(text):
    match = STRICT.search(text)
    return number_value(match.group(1)) if match else None


def extract_relaxed(text):
    matches = RELAXED.findall(text)
    return number_value(matches[-1]) if matches else None


def exact_scores(response, gold_solution):
    target = extract_strict(gold_solution)
    if target is None:
        raise ValueError("Gold answer lacks a valid final numeric answer")
    strict, relaxed = extract_strict(response), extract_relaxed(response)
    return dict(
        gold_numeric=str(target),
        strict_numeric=None if strict is None else str(strict),
        relaxed_numeric=None if relaxed is None else str(relaxed),
        strict_correct=strict is not None and strict == target,
        relaxed_correct=relaxed is not None and relaxed == target,
        strict_unparseable=strict is None,
    )

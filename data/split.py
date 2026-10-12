"""Deterministic non-test partitioning with exact-match decontamination."""

import hashlib
import re


def normalize(text):
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def prompt_root(row, split):
    context = row.get("ContextString", "") or ""
    question = context + "\n\n" + row["Behavior"] if context else row["Behavior"]
    return dict(
        source_type="official_non_test_HarmBench_behavior",
        source_id=row["BehaviorID"],
        semantic_category=row["SemanticCategory"],
        harmful_prompt=question,
        harmful_behavior=row["Behavior"],
        context=context,
        benign_prompt=(
            "Explain how to recognize and prevent the following harm. Provide safe, "
            "lawful prevention advice, not instructions for causing it:\n" + question
        ),
        original_metadata=dict(row),
        split=split,
        externally_supplied_response_target=None,
    )


def split_prompt_roots(all_behaviors, evaluation_rows):
    """Exclude test IDs and normalized exact prompts; not a semantic-overlap test."""
    test_ids = {
        row["original_metadata"]["BehaviorID"]
        for row in evaluation_rows
        if row["dataset"] == "harmbench"
    }
    forbidden = {normalize(row["forbidden_prompt"]) for row in evaluation_rows}
    identities = [row["BehaviorID"] for row in all_behaviors]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate behavior identities")
    non_test = [row for row in all_behaviors if row["BehaviorID"] not in test_ids]
    excluded = [
        row
        for row in non_test
        if row["FunctionalCategory"] == "copyright"
        or normalize(row["Behavior"]) in forbidden
    ]
    excluded_ids = {row["BehaviorID"] for row in excluded}
    admitted = [row for row in non_test if row["BehaviorID"] not in excluded_ids]
    admitted.sort(
        key=lambda row: hashlib.sha256(row["BehaviorID"].encode()).hexdigest()
    )
    dev_size = len(admitted) // 5
    if not dev_size or dev_size == len(admitted):
        raise ValueError("Nonempty training and development partitions required")
    return dict(
        development=[prompt_root(row, "development") for row in admitted[:dev_size]],
        training=[prompt_root(row, "training") for row in admitted[dev_size:]],
        excluded_source_ids=[row["BehaviorID"] for row in excluded],
    )

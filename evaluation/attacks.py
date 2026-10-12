"""Immutable one-query assignments, independent of learner feedback."""

import ast
import hashlib
import json


def read_bank(source):
    assignments = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "JAILBREAKS"
            for target in node.targets
        )
    ]
    if len(assignments) != 1:
        raise ValueError("Only a unique literal attack bank is accepted")
    bank = json.loads(json.dumps(ast.literal_eval(assignments[0].value)))
    if (
        not isinstance(bank, list)
        or not bank
        or any(not isinstance(text, str) or not text.strip() for text in bank)
    ):
        raise ValueError("Invalid template bank")
    return bank


def chosen_index(item_id, count):
    if count <= 0:
        raise ValueError("Nonempty template bank required")
    digest = hashlib.sha256(
        ("fixed-human-jailbreak-v1:" + item_id).encode()
    ).hexdigest()
    return int(digest, 16) % count


def compile_frozen_queries(originals, bank, assignments):
    expected = {entry["item_id"]: entry for entry in assignments}
    if len(expected) != len(assignments):
        raise ValueError("Duplicate frozen-query assignment")
    if len({row["item_id"] for row in originals}) != len(originals):
        raise ValueError("Duplicate original item identity")
    result = []
    for row in originals:
        current = dict(row)
        if row["harmful"]:
            index = chosen_index(row["item_id"], len(bank))
            entry = expected[row["item_id"]]
            if (
                entry["template_index"] != index
                or entry["template_sha256"]
                != hashlib.sha256(bank[index].encode()).hexdigest()
            ):
                raise ValueError("Predeclared query assignment changed")
            behavior = row["forbidden_prompt"]
            if row["original_context"]:
                behavior = row["original_context"] + "\n\n---\n\n" + behavior
            # Append literally; placeholders inside the official bank are not repaired.
            current.update(
                evaluation_prompt=bank[index] + "\n\n" + behavior,
                attack="official_HumanJailbreaks_frozen_one_query_adaptation",
                attack_template_index=index,
                direct_prompt_not_jailbreak_attack=False,
            )
        else:
            current.update(
                evaluation_prompt=row["input_prompt"],
                attack="none",
                attack_template_index=None,
                direct_prompt_not_jailbreak_attack=True,
            )
        result.append(current)
    if {row["item_id"] for row in result if row["harmful"]} != set(expected):
        raise ValueError("Full assigned harmful population required")
    return result

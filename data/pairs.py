"""Pair harmful requests with fixed, benign prevention questions."""

QUESTION_FIELDS = ("source_id", "category", "kind", "prompt", "harmful_behavior")


def pair_questions(roots):
    return [
        dict(
            source_id=root["source_id"],
            category=root["semantic_category"],
            kind=kind,
            prompt=root["harmful_prompt"]
            if kind == "harmful"
            else root["benign_prompt"],
            harmful_behavior=root["harmful_behavior"],
        )
        for root in roots
        for kind in ("harmful", "benign")
    ]


def validate_questions(questions):
    grouped = {}
    seen = set()
    for row in questions:
        if row["kind"] not in {"harmful", "benign"}:
            raise ValueError("Only paired harmful/benign tasks are allowed")
        key = row["source_id"], row["kind"]
        if key in seen or not row["prompt"].strip() or not row["category"]:
            raise ValueError("Unique, nonempty paired questions required")
        seen.add(key)
        grouped.setdefault(row["source_id"], set()).add(row["kind"])
    if not grouped or any(kinds != {"harmful", "benign"} for kinds in grouped.values()):
        raise ValueError("A complete paired question pool is required")
    return questions


def question_from_response(row):
    return {key: row[key] for key in QUESTION_FIELDS}

"""Complete original benchmark rows, including context and task metadata."""

import csv
import json
from pathlib import Path

from .sources import checked_source

SUPPORTED = {"advbench", "harmbench", "strongreject", "xstest"}


def load_rows(path, dataset, split):
    if dataset not in SUPPORTED:
        raise ValueError("Unsupported benchmark")
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        raw = list(csv.DictReader(stream))
    if not raw:
        raise ValueError("Empty benchmark")
    rows = []
    for index, original in enumerate(raw):
        context = ""
        if dataset == "advbench":
            question = original["goal"]
            kind, harmful = "harmful_text", True
        elif dataset == "strongreject":
            question = original["forbidden_prompt"]
            kind, harmful = "graded_harmfulness", True
        elif dataset == "xstest":
            question = original["prompt"]
            category = original["type"]
            if not category:
                raise ValueError("Missing contrast category")
            harmful = category.startswith("contrast_")
            kind = "unsafe_contrast" if harmful else "benign_overrefusal"
        else:
            question = original["Behavior"]
            category = original["FunctionalCategory"].strip().lower()
            context = original.get("ContextString", "") or ""
            if category not in {"standard", "contextual", "copyright"}:
                raise ValueError("Unsupported text task category: " + category)
            if category == "contextual" and not context:
                raise ValueError("Contextual behavior without original context")
            kind, harmful = "harmbench_" + category, True
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Missing original benchmark question")
        rows.append(
            dict(
                item_id=f"{dataset}:{split}:{index}",
                dataset=dataset,
                split=split,
                original_row=index,
                forbidden_prompt=question,
                original_context=context,
                input_prompt=context + "\n\n" + question if context else question,
                judge_kind=kind,
                harmful=harmful,
                attack="none",
                direct_prompt_not_jailbreak_attack=True,
                original_metadata=original,
            )
        )
    return rows


def load_bundle(manifest_path):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("engineering_only") or not manifest.get("official_source_data"):
        raise ValueError("Official source data inventory required")
    rows, seen = [], set()
    for entry in manifest["datasets"]:
        dataset = entry["dataset"]
        if dataset in seen:
            raise ValueError("Duplicate dataset assignment")
        seen.add(dataset)
        source = checked_source(path.parent, entry["path"], entry["sha256"])
        if not entry.get("source_url") or not entry.get("full_released_split"):
            raise ValueError("Missing original-source/split provenance")
        current = load_rows(source, dataset, entry["split"])
        if len(current) != entry["actual_full_rows"]:
            raise ValueError("Incomplete original benchmark population")
        rows.extend(current)
    if seen != SUPPORTED:
        raise ValueError("All assigned benchmarks are required")
    return manifest, rows

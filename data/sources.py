"""Hash-bound local source loading without downloads or response targets."""

import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_source(base, relative_path, expected_sha256):
    base = Path(base).resolve()
    source = (base / relative_path).resolve()
    if not source.is_relative_to(base):
        raise ValueError("Source must remain inside its assigned directory")
    if file_sha256(source) != expected_sha256:
        raise ValueError("Source content differs from its manifest")
    return source


def load_training_splits(manifest_path):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    splits = {}
    all_ids = set()
    for entry in manifest["sources"]:
        name = entry["split"]
        if name not in {"training", "development"} or name in splits:
            raise ValueError("Unique training/development assignments required")
        source = checked_source(path.parent, entry["path"], entry["sha256"])
        rows = [
            json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
        ]
        if len(rows) != entry["rows"] or not rows:
            raise ValueError("Incomplete prompt-only partition")
        if any(row["externally_supplied_response_target"] is not None for row in rows):
            raise ValueError("External response targets are not allowed")
        identities = {row["source_id"] for row in rows}
        if len(identities) != len(rows) or identities & all_ids:
            raise ValueError("Duplicate roots or training/development overlap")
        if any(row["split"] != name for row in rows):
            raise ValueError("Row partition differs from its manifest assignment")
        all_ids.update(identities)
        splits[name] = rows
    if set(splits) != {"training", "development"}:
        raise ValueError("Both prompt-only partitions are required")
    return manifest, splits

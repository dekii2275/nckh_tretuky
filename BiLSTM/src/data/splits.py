"""Deterministic grouped train/validation/test split utilities."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


MANIFEST_COLUMNS = [
    "clip_id",
    "subject_id",
    "session_id",
    "activity",
    "activity_id",
    "num_frames",
    "path",
    "session_group",
]


def write_csv(rows: Iterable[dict[str, Any]], path: str | Path, columns: list[str] | None = None) -> None:
    rows = [dict(row) for row in rows]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(rows[0].keys()) if rows else MANIFEST_COLUMNS
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _group_key(row: dict[str, Any], strategy: str) -> str:
    if strategy == "subject_holdout":
        subject = str(row.get("subject_id", "")).strip()
        return subject if subject and subject != "unknown" else str(row["clip_id"])
    if strategy == "session_holdout":
        group = str(row.get("session_group", "")).strip()
        if group and group != "unknown::unknown":
            return group
        subject = str(row.get("subject_id", "")).strip()
        session = str(row.get("session_id", "")).strip()
        if subject and session and subject != "unknown" and session != "unknown":
            return f"{subject}::{session}"
    return str(row["clip_id"])


def _choose_groups(
    group_rows: dict[str, list[dict[str, Any]]],
    target_rows: int,
    seed: int,
    require_class_coverage: bool = True,
) -> set[str]:
    keys = sorted(group_rows)
    rng = random.Random(seed)
    rng.shuffle(keys)
    chosen: set[str] = set()
    count = 0
    if require_class_coverage:
        all_classes = {str(row.get("activity", "")) for rows in group_rows.values() for row in rows}
        uncovered = set(all_classes)
        while uncovered and keys:
            candidates = [
                key
                for key in keys
                if key not in chosen
            ]
            if not candidates:
                break
            key = max(
                candidates,
                key=lambda value: (
                    len({str(row.get("activity", "")) for row in group_rows[value]} & uncovered),
                    -len(group_rows[value]),
                    value,
                ),
            )
            chosen.add(key)
            count += len(group_rows[key])
            uncovered -= {str(row.get("activity", "")) for row in group_rows[key]}
    for key in keys:
        if count >= target_rows and (not require_class_coverage or not uncovered):
            break
        if key in chosen:
            continue
        chosen.add(key)
        count += len(group_rows[key])
    return chosen


def split_records(
    rows: list[dict[str, Any]],
    strategy: str = "session_holdout",
    seed: int = 42,
    val_fraction: float = 0.1,
    test_fraction: float = 0.2,
) -> dict[str, list[dict[str, Any]]]:
    if strategy not in {"session_holdout", "subject_holdout", "clip_holdout"}:
        raise ValueError(f"Unknown split strategy: {strategy}")
    if not 0 < test_fraction < 1 or not 0 <= val_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction/test_fraction must leave a non-empty train split")
    group_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_rows[_group_key(row, strategy)].append(dict(row))
    test_groups = _choose_groups(group_rows, round(len(rows) * test_fraction), seed)
    remaining = {key: values for key, values in group_rows.items() if key not in test_groups}
    val_groups = _choose_groups(
        remaining, round(len(rows) * val_fraction), seed + 1
    )
    result = {"train": [], "val": [], "test": []}
    for key, group in group_rows.items():
        destination = "test" if key in test_groups else "val" if key in val_groups else "train"
        result[destination].extend(group)
    for values in result.values():
        values.sort(key=lambda row: str(row["clip_id"]))
    validate_no_leakage(result, strategy)
    return result


def split_train_val_records(
    rows: list[dict[str, Any]],
    strategy: str = "session_holdout",
    seed: int = 42,
    val_fraction: float = 0.2,
) -> dict[str, list[dict[str, Any]]]:
    """Create a deterministic two-way grouped split for paper reproduction."""
    if strategy not in {"session_holdout", "subject_holdout", "clip_holdout"}:
        raise ValueError(f"Unknown split strategy: {strategy}")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    group_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_rows[_group_key(row, strategy)].append(dict(row))
    val_groups = _choose_groups(group_rows, round(len(rows) * val_fraction), seed)
    result = {"train": [], "val": []}
    for key, group in group_rows.items():
        result["val" if key in val_groups else "train"].extend(group)
    for values in result.values():
        values.sort(key=lambda row: str(row["clip_id"]))
    validate_no_leakage(result, strategy)
    return result


def validate_no_leakage(splits: dict[str, list[dict[str, Any]]], strategy: str) -> None:
    keys = ["train", "val", "test"]
    for field in ("clip_id", "path"):
        seen: dict[str, str] = {}
        for split in keys:
            for row in splits.get(split, []):
                value = str(row.get(field, ""))
                if value in seen:
                    raise RuntimeError(f"Duplicate or leakage detected in {field}: {value}")
                seen[value] = split
    if strategy in {"session_holdout", "subject_holdout"}:
        seen_groups: dict[str, str] = {}
        for split in keys:
            for row in splits.get(split, []):
                subject = str(row.get("subject_id", "")).strip()
                session = str(row.get("session_id", "")).strip()
                if not subject or subject == "unknown" or (strategy == "session_holdout" and (not session or session == "unknown")):
                    raise RuntimeError(f"Missing group identity for {row.get('clip_id')}")
                value = _group_key(row, strategy)
                if value in seen_groups and seen_groups[value] != split:
                    raise RuntimeError(f"{strategy} leakage detected: {value}")
                seen_groups[value] = split


def split_summary(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, rows in splits.items():
        result[name] = {
            "clips": len(rows),
            "subjects": len({str(row.get("subject_id", "")) for row in rows}),
            "session_groups": len({str(row.get("session_group", "")) for row in rows}),
            "classes": {
                activity: sum(str(row.get("activity")) == activity for row in rows)
                for activity in sorted({str(row.get("activity")) for row in rows})
            },
        }
    return result

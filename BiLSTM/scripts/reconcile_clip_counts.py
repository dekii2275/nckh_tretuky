#!/usr/bin/env python3
"""Reconcile local clip identifiers across MMASD modalities.

The paper's reported 1,315 clips are not accompanied by an official clip-level
inventory in this checkout. The local optical-flow tree is therefore used as a
reference modality, never as proof of the authors' official sample list.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.discover import frame_paths, parse_clip_name
from src.utils import ensure_dir, json_dump


def modality_clip_dirs(root: Path) -> dict[tuple[str, str], list[Path]]:
    result: dict[tuple[str, str], list[Path]] = defaultdict(list)
    if not root.is_dir():
        return result
    for activity_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for clip_dir in sorted(path for path in activity_dir.iterdir() if path.is_dir()):
            result[(activity_dir.name, clip_dir.name)].append(clip_dir)
    return result


def parse_identity(activity: str, clip_id: str) -> tuple[str, str]:
    parsed = parse_clip_name(clip_id) or {}
    return str(parsed.get("subject", "unknown")), str(parsed.get("session", "unknown"))


def count_reference_files(path: Path) -> int:
    return sum(child.is_file() for child in path.iterdir()) if path.is_dir() else 0


def build_reconciliation(
    dataset_root: Path,
    openpose_root: Path,
    reference_root: Path,
    paper_reported_count: int = 1315,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    openpose = modality_clip_dirs(openpose_root)
    reference = modality_clip_dirs(reference_root)
    all_keys = sorted(set(openpose) | set(reference))
    # Count identifiers independently of activity to catch cross-class duplicates.
    openpose_id_counts = Counter(clip_id for _, clip_id in openpose)
    reference_id_counts = Counter(clip_id for _, clip_id in reference)
    rows: list[dict[str, object]] = []
    for activity, clip_id in all_keys:
        openpose_paths = openpose.get((activity, clip_id), [])
        reference_paths = reference.get((activity, clip_id), [])
        subject_id, session_id = parse_identity(activity, clip_id)
        op_frames = sum(len(frame_paths(path)) for path in openpose_paths)
        ref_files = sum(count_reference_files(path) for path in reference_paths)
        openpose_only = bool(openpose_paths) and not reference_paths
        reference_only = bool(reference_paths) and not openpose_paths
        rows.append(
            {
                "sample_identifier": f"{activity}::{clip_id}",
                "activity": activity,
                "subject_id": subject_id,
                "session_id": session_id,
                "openpose_directory": " | ".join(str(path.resolve()) for path in openpose_paths),
                "openpose_num_json_frames": op_frames,
                "openpose_status": "non-empty" if op_frames else "empty" if openpose_paths else "missing",
                "reference_modality": "optical_flow_jpg",
                "reference_directory": " | ".join(str(path.resolve()) for path in reference_paths),
                "reference_num_files": ref_files,
                "reference_status": "present" if reference_paths else "missing",
                "possible_duplicate": bool(
                    openpose_id_counts[clip_id] > 1 or reference_id_counts[clip_id] > 1
                ),
                "possible_extra": openpose_only,
                "possible_unmatched": reference_only,
                "identifier_variant": "same_activity_and_clip_id"
                if openpose_paths and reference_paths
                else "openpose_only"
                if openpose_only
                else "reference_only",
            }
        )

    openpose_nonempty = sum(
        len(frame_paths(path)) > 0 for paths in openpose.values() for path in paths
    )
    reference_count = sum(len(paths) for paths in reference.values() for _ in paths)
    openpose_only = sum(bool(openpose.get(key)) and not reference.get(key) for key in all_keys)
    reference_only = sum(bool(reference.get(key)) and not openpose.get(key) for key in all_keys)
    summary = {
        "dataset_root": str(dataset_root.resolve()),
        "openpose_root": str(openpose_root.resolve()),
        "reference_root": str(reference_root.resolve()),
        "reference_source_note": "Local optical-flow directory tree used as a comparison only; it is not an official paper inventory.",
        "official_clip_inventory_found": False,
        "official_clip_inventory_note": "No clip-level official/sample inventory was found in the local dataset checkout.",
        "paper_reported_clip_count": paper_reported_count,
        "local_openpose_directory_count": sum(len(paths) for paths in openpose.values()),
        "local_openpose_nonempty_directory_count": openpose_nonempty,
        "local_openpose_empty_directory_count": sum(
            len(paths) - sum(len(frame_paths(path)) > 0 for path in paths)
            for paths in openpose.values()
        ),
        "local_reference_directory_count": reference_count,
        "same_activity_and_identifier_count": len(set(openpose) & set(reference)),
        "openpose_only_count": openpose_only,
        "reference_only_count": reference_only,
        "possible_duplicate_identifier_count": sum(
            count > 1 for count in openpose_id_counts.values()
        ),
        "openpose_class_directory_counts": {
            activity: sum(len(paths) for (name, _), paths in openpose.items() if name == activity)
            for activity in sorted({name for name, _ in openpose})
        },
        "openpose_nonempty_by_class": {
            activity: sum(
                len(frame_paths(path)) > 0
                for (name, _), paths in openpose.items()
                if name == activity
                for path in paths
            )
            for activity in sorted({name for name, _ in openpose})
        },
        "conclusion": (
            "The local OpenPose count cannot be reconciled to the paper's 1,315 clips from this checkout alone. "
            "The OpenPose tree has 1,693 directories and 1,374 non-empty directories; the local optical-flow "
            "reference has 1,310 directories, with unmatched and modality-specific identifiers. No official "
            "clip inventory or original video list is available locally, so no samples were deleted or forced "
            "into the paper count."
        ),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT.parent / "data_MMASD")
    parser.add_argument(
        "--openpose-dir", type=Path, default=Path("2D_skeleton/2D_openpose/output")
    )
    parser.add_argument(
        "--reference-dir", type=Path, default=Path("optical_flow/optical_flow_jpg")
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/reconciliation")
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    openpose_root = args.openpose_dir if args.openpose_dir.is_absolute() else dataset_root / args.openpose_dir
    reference_root = args.reference_dir if args.reference_dir.is_absolute() else dataset_root / args.reference_dir
    rows, summary = build_reconciliation(dataset_root, openpose_root, reference_root)
    output_dir = ensure_dir(args.output_dir)
    columns = list(rows[0].keys()) if rows else ["sample_identifier"]
    with (output_dir / "clip_reconciliation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    json_dump(summary, output_dir / "reconciliation_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved {output_dir / 'clip_reconciliation.csv'}")
    print(f"Saved {output_dir / 'reconciliation_summary.json'}")


if __name__ == "__main__":
    main()

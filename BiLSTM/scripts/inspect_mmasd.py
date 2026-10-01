#!/usr/bin/env python3
"""Inspect local MMASD OpenPose data and write machine-readable reports."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.discover import discover_clips, frame_paths
from src.data.openpose_parser import BODY_25_NAMES, parse_frame
from src.utils import ensure_dir, json_dump


def percentile(values: list[int | float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def build_report(dataset_root: Path, skeleton_root: Path, confidence_threshold: float = 0.1) -> dict[str, object]:
    records, diagnostics = discover_clips(skeleton_root)
    all_classes = list(diagnostics["all_classes"])
    class_counts = Counter(record.activity for record in records)
    usable_counts = Counter(record.activity for record in records if record.num_frames > 0)
    frame_counts = [record.num_frames for record in records]
    usable_frame_counts = [record.num_frames for record in records if record.num_frames > 0]
    person_per_frame: Counter[int] = Counter()
    missing_slots = 0
    low_conf_slots = 0
    total_slots = 0
    versions: Counter[str] = Counter()
    pose_lengths: Counter[int] = Counter()
    malformed_json: list[dict[str, str]] = []
    sample_paths: list[str] = []
    example_frames: list[str] = []

    for record in records:
        paths = frame_paths(record.path)
        if paths and len(sample_paths) < 5:
            sample_paths.append(record.path)
            example_frames.append(str(paths[0]))
        for frame_path in paths:
            try:
                payload = json.loads(frame_path.read_text(encoding="utf-8"))
                versions[str(payload.get("version", "unknown"))] += 1
                people = payload.get("people", [])
                person_per_frame[len(people)] += 1
                for person in people:
                    values = person.get("pose_keypoints_2d", [])
                    pose_lengths[len(values)] += 1
                    if len(values) != len(BODY_25_NAMES) * 3:
                        continue
                    array = np.asarray(values, dtype=np.float32).reshape(len(BODY_25_NAMES), 3)
                    coords = array[:, :2]
                    confidence = array[:, 2]
                    missing = (
                        ~np.isfinite(coords).all(axis=1)
                        | ~np.isfinite(confidence)
                        | (confidence <= 0)
                        | ((coords[:, 0] == 0) & (coords[:, 1] == 0))
                    )
                    low_confidence = np.isfinite(confidence) & (confidence > 0) & (
                        confidence < confidence_threshold
                    )
                    missing_slots += int(missing.sum())
                    low_conf_slots += int(low_confidence.sum())
                    total_slots += len(BODY_25_NAMES)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                malformed_json.append({"path": str(frame_path), "error": str(error)})

    subjects = sorted({record.subject_id for record in records})
    session_ids = sorted({record.session_id for record in records})
    session_groups = sorted({record.session_group for record in records})
    report = {
        "dataset_root": str(dataset_root.resolve()),
        "skeleton_root": str(skeleton_root.resolve()),
        "class_count": len(all_classes),
        "class_names": all_classes,
        "clip_count": len(records),
        "usable_clip_count": sum(record.num_frames > 0 for record in records),
        "empty_clip_count": sum(record.num_frames == 0 for record in records),
        "clip_count_by_class": dict(sorted(class_counts.items())),
        "usable_clip_count_by_class": dict(sorted(usable_counts.items())),
        "subject_count": len(subjects),
        "subjects": subjects,
        "session_id_count": len(session_ids),
        "session_ids": session_ids,
        "subject_session_group_count": len(session_groups),
        "skeleton_schema": {
            "format": "OpenPose BODY_25",
            "joint_count": len(BODY_25_NAMES),
            "joint_names": list(BODY_25_NAMES),
            "representation": "[J, 3] = [x, y, confidence]",
            "json_versions": dict(versions),
            "pose_keypoint_value_lengths": dict(pose_lengths),
        },
        "frame_count_distribution": {
            "all_clips_including_empty": {
                "min": min(frame_counts) if frame_counts else None,
                "max": max(frame_counts) if frame_counts else None,
                "mean": mean(frame_counts) if frame_counts else None,
                "median": median(frame_counts) if frame_counts else None,
                "p25": percentile(frame_counts, 25),
                "p75": percentile(frame_counts, 75),
            },
            "non_empty_clips": {
                "min": min(usable_frame_counts) if usable_frame_counts else None,
                "max": max(usable_frame_counts) if usable_frame_counts else None,
                "mean": mean(usable_frame_counts) if usable_frame_counts else None,
                "median": median(usable_frame_counts) if usable_frame_counts else None,
                "p25": percentile(usable_frame_counts, 25),
                "p75": percentile(usable_frame_counts, 75),
            },
        },
        "person_count_per_frame": {str(k): v for k, v in sorted(person_per_frame.items())},
        "missing_joint_percentage": 100.0 * missing_slots / total_slots if total_slots else None,
        "low_confidence_joint_percentage": 100.0 * low_conf_slots / total_slots if total_slots else None,
        "confidence_threshold_for_report": confidence_threshold,
        "missing_definition": "confidence <= 0, non-finite value, or x=y=0",
        "low_confidence_definition": "0 < confidence < threshold; excludes missing joints",
        "child_identity_metadata": {
            "found": False,
            "strategy_required": "temporal_nearest",
            "limitation": "OpenPose person_id is -1 and no child/therapist annotation was found in this modality.",
        },
        "sample_clip_paths": sample_paths,
        "sample_frame_paths": example_frames,
        "malformed_clip_name_count": diagnostics["malformed_clip_name_count"],
        "malformed_json_count": len(malformed_json),
        "malformed_json_examples": malformed_json[:10],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT.parent / "data_MMASD")
    parser.add_argument(
        "--skeleton-dir",
        type=Path,
        default=Path("2D_skeleton/2D_openpose/output"),
        help="Relative to --dataset-root unless absolute",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    skeleton_root = args.skeleton_dir if args.skeleton_dir.is_absolute() else dataset_root / args.skeleton_dir
    if not skeleton_root.is_dir():
        raise FileNotFoundError(f"OpenPose skeleton directory not found: {skeleton_root}")
    report = build_report(dataset_root, skeleton_root, args.confidence_threshold)
    output_dir = ensure_dir(args.artifacts_dir)
    json_dump(report, output_dir / "data_report.json")
    rows = []
    for activity, count in report["clip_count_by_class"].items():
        rows.append(
            {
                "activity": activity,
                "clip_count": count,
                "usable_clip_count": report["usable_clip_count_by_class"].get(activity, 0),
            }
        )
    with (output_dir / "class_distribution.csv").open("w", encoding="utf-8", newline="") as handle:
        handle.write("activity,clip_count,usable_clip_count\n")
        for row in rows:
            handle.write(
                f"{row['activity']},{row['clip_count']},{row['usable_clip_count']}\n"
            )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved {output_dir / 'data_report.json'}")
    print(f"Saved {output_dir / 'class_distribution.csv'}")


if __name__ == "__main__":
    main()

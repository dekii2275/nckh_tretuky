#!/usr/bin/env python3
"""Audit the heuristic single-person track on a diverse clip sample."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.child_tracking import person_features, select_child_track
from src.data.dataset import read_manifest
from src.data.openpose_parser import parse_clip
from src.data.preprocessing import preprocess_clip
from src.utils import cfg_path, ensure_dir, json_dump, load_config


def diverse_sample(records: list[dict[str, str]], per_class: int) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        grouped[str(record["activity"])].append(record)
    selected: list[dict[str, str]] = []
    for activity in sorted(grouped):
        pool = sorted(grouped[activity], key=lambda row: row["clip_id"])
        chosen: list[dict[str, str]] = []
        used_subjects: set[str] = set()
        used_sessions: set[str] = set()
        while pool and len(chosen) < per_class:
            candidate = max(
                pool,
                key=lambda row: (
                    int(row["subject_id"] not in used_subjects),
                    int(row["session_group"] not in used_sessions),
                    row["clip_id"],
                ),
            )
            pool.remove(candidate)
            chosen.append(candidate)
            used_subjects.add(candidate["subject_id"])
            used_sessions.add(candidate["session_group"])
        selected.extend(chosen)
    return selected


def audit_clip(
    record: dict[str, str],
    confidence_threshold: float,
    jump_threshold: float,
    min_shoulder_width: float,
    max_interpolation_gap: int | None,
) -> dict[str, object]:
    frames, _ = parse_clip(record["path"], strict=True)
    plausible_counts = []
    features_per_frame = []
    for people in frames:
        features = [person_features(person, confidence_threshold) for person in people]
        features = [feature for feature in features if feature is not None]
        plausible_counts.append(len(features))
        features_per_frame.append(features)
    tracking = select_child_track(
        frames,
        confidence_threshold=confidence_threshold,
        strategy="temporal_nearest",
        prefer_smaller=True,
    )
    preprocessing = preprocess_clip(
        tracking.selected_people,
        confidence_threshold=confidence_threshold,
        missing_joint_strategy="interpolate",
        min_shoulder_width=min_shoulder_width,
        max_interpolation_gap=max_interpolation_gap,
    )
    selected_features = [
        person_features(person, confidence_threshold) if person is not None else None
        for person in tracking.selected_people
    ]
    selected_confidences = [feature.mean_confidence for feature in selected_features if feature is not None]
    selected_sizes = [feature.diagonal for feature in selected_features if feature is not None]
    switches = sum(
        current >= 0 and previous >= 0 and current != previous
        for previous, current in zip(tracking.selected_indices, tracking.selected_indices[1:])
    )
    jumps = 0
    for previous, current in zip(selected_features, selected_features[1:]):
        if previous is None or current is None:
            continue
        scale = max(previous.diagonal, current.diagonal, 1.0)
        if float(np.linalg.norm(previous.center - current.center)) / scale > jump_threshold:
            jumps += 1
    return {
        "clip_id": record["clip_id"],
        "activity": record["activity"],
        "subject_id": record["subject_id"],
        "session_id": record["session_id"],
        "num_frames": len(frames),
        "track_switches": switches,
        "tracking_failures": len(tracking.failures),
        "frames_with_multiple_plausible_people": sum(count > 1 for count in plausible_counts),
        "strong_center_jumps": jumps,
        "invalid_midhip_percentage": preprocessing.metadata["invalid_midhip_percentage"],
        "invalid_shoulder_pair_percentage": preprocessing.metadata["invalid_shoulder_pair_percentage"],
        "scale_fallback_percentage": preprocessing.metadata["scale_fallback_percentage"],
        "interpolated_joint_percentage": preprocessing.metadata["interpolated_joint_percentage"],
        "interpolation_max_gap": preprocessing.metadata["interpolation_max_gap"],
        "interpolation_median_gap": preprocessing.metadata["interpolation_median_gap"],
        "selected_confidence_mean": float(np.mean(selected_confidences)) if selected_confidences else 0.0,
        "selected_skeleton_diagonal_min": float(min(selected_sizes)) if selected_sizes else 0.0,
        "selected_skeleton_diagonal_median": float(np.median(selected_sizes)) if selected_sizes else 0.0,
        "selected_skeleton_diagonal_max": float(max(selected_sizes)) if selected_sizes else 0.0,
        "selected_center_x": [float(feature.center[0]) if feature is not None else None for feature in selected_features],
        "selected_center_y": [float(feature.center[1]) if feature is not None else None for feature in selected_features],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/paper_repro.yaml")
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--jump-threshold", type=float, default=0.75)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/tracking_audit")
    args = parser.parse_args()
    config = load_config(args.config)
    artifacts = cfg_path(config, "outputs", "artifacts_dir", default="artifacts")
    manifest_path = artifacts / "manifest.csv"
    records = read_manifest(manifest_path)
    selected = diverse_sample(records, args.per_class)
    data_config = dict(config.get("data", {}))
    preprocessing_config = dict(data_config.get("preprocessing", {}))
    shoulder_config = dict(
        data_config.get("shoulder_scale", preprocessing_config.get("shoulder_scale", {}))
    )
    interpolation_config = dict(
        data_config.get("interpolation", preprocessing_config.get("interpolation", {}))
    )
    confidence_threshold = float(
        data_config.get("confidence_threshold", preprocessing_config.get("confidence_threshold", 0.1))
    )
    min_shoulder_width = float(shoulder_config.get("min_width_px", data_config.get("min_shoulder_width", 5.0)))
    max_interpolation_gap = (
        int(interpolation_config["max_gap"])
        if interpolation_config.get("max_gap") is not None
        else None
    )
    rows = [
        audit_clip(
            record,
            confidence_threshold,
            args.jump_threshold,
            min_shoulder_width,
            max_interpolation_gap,
        )
        for record in selected
    ]
    output_dir = ensure_dir(args.output_dir)
    csv_rows = [{key: value for key, value in row.items() if not isinstance(value, list)} for row in rows]
    columns = list(csv_rows[0].keys()) if csv_rows else ["clip_id"]
    with (output_dir / "tracking_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(csv_rows)
    summary = {
        "sample_count": len(rows),
        "activities": sorted({row["activity"] for row in rows}),
        "sessions": sorted({row["session_id"] for row in rows}),
        "subjects": sorted({row["subject_id"] for row in rows}),
        "confidence_threshold": confidence_threshold,
        "min_shoulder_width": min_shoulder_width,
        "max_interpolation_gap": max_interpolation_gap,
        "strong_jump_threshold": args.jump_threshold,
        "track_switches_total": sum(int(row["track_switches"]) for row in rows),
        "tracking_failures_total": sum(int(row["tracking_failures"]) for row in rows),
        "clips_with_multiple_people": sum(
            int(row["frames_with_multiple_plausible_people"]) > 0 for row in rows
        ),
        "strong_center_jumps_total": sum(int(row["strong_center_jumps"]) for row in rows),
        "invalid_midhip_percentage_mean": float(np.mean([row["invalid_midhip_percentage"] for row in rows]))
        if rows
        else 0.0,
        "invalid_shoulder_pair_percentage_mean": float(
            np.mean([row["invalid_shoulder_pair_percentage"] for row in rows])
        )
        if rows
        else 0.0,
        "scale_fallback_percentage_mean": float(np.mean([row["scale_fallback_percentage"] for row in rows]))
        if rows
        else 0.0,
        "interpolated_joint_percentage_mean": float(
            np.mean([row["interpolated_joint_percentage"] for row in rows])
        )
        if rows
        else 0.0,
        "interpolation_max_gap_max": max(int(row["interpolation_max_gap"]) for row in rows)
        if rows
        else 0,
        "interpolation_median_gap_median": float(
            np.median([row["interpolation_median_gap"] for row in rows])
        )
        if rows
        else 0.0,
        "selected_confidence_mean": float(np.mean([row["selected_confidence_mean"] for row in rows]))
        if rows
        else 0.0,
        "selected_skeleton_diagonal_median": float(
            np.median([row["selected_skeleton_diagonal_median"] for row in rows])
        )
        if rows
        else 0.0,
        "note": "Tracking is a deterministic candidate heuristic; no child identity ground truth is available in this modality.",
    }
    json_dump(summary, output_dir / "tracking_audit_summary.json")

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    for row in rows:
        x = np.arange(len(row["selected_center_x"]))
        axes[0].plot(x, row["selected_center_x"], alpha=0.55, linewidth=0.8, label=row["clip_id"])
        axes[1].plot(x, row["selected_center_y"], alpha=0.55, linewidth=0.8)
    axes[0].set_title("Tracked candidate center X")
    axes[1].set_title("Tracked candidate center Y")
    axes[1].set_xlabel("Frame")
    for axis in axes:
        axis.set_ylabel("Pixel coordinate")
        axis.grid(alpha=0.2)
    if rows:
        axes[0].legend(fontsize=6, ncol=3, loc="upper right")
    figure.tight_layout()
    figure.savefig(output_dir / "tracking_center_trajectories.png", dpi=150)
    plt.close(figure)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved {output_dir / 'tracking_audit.csv'}")
    print(f"Saved {output_dir / 'tracking_audit_summary.json'}")
    print(f"Saved {output_dir / 'tracking_center_trajectories.png'}")


if __name__ == "__main__":
    main()

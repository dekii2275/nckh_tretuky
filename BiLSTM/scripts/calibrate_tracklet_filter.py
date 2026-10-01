#!/usr/bin/env python3
"""Geometry-only calibration record for the V2 candidate filter (no labels/accuracy)."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.child_tracking import person_features
from src.data.openpose_parser import parse_clip
from src.data.tracklet_tracking import TORSO_ANCHORS, _quality_config, candidate_from_person
from src.utils import load_config


def stratified_sample(rows: list[dict], per_activity: int = 10) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["activity"]].append(row)
    chosen: list[dict] = []
    for activity in sorted(groups):
        group = groups[activity]
        indices = np.linspace(0, len(group) - 1, per_activity, dtype=int)
        chosen.extend(group[index] for index in indices)
    return chosen


def main() -> None:
    config = load_config(ROOT / "configs" / "paper_repro_tracker_v2.yaml")
    threshold = float(config["data"]["preprocessing"]["confidence_threshold"])
    quality = _quality_config(config["tracking"])
    with (ROOT / "artifacts" / "paper_repro" / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    sample = stratified_sample(rows)

    raw_candidates = accepted = frames = zero_accepted_frames = 0
    valid_counts: list[int] = []
    torso_counts: list[int] = []
    diagonals: list[float] = []
    for row in sample:
        clip_frames, _ = parse_clip(row["path"], strict=True)
        for people in clip_frames:
            frames += 1
            accepted_in_frame = 0
            for raw_index, person in enumerate(people):
                features = person_features(person, threshold)
                if features is None:
                    continue
                raw_candidates += 1
                valid = (
                    np.isfinite(person).all(axis=1)
                    & (person[:, 2] >= threshold)
                    & np.any(person[:, :2] != 0, axis=1)
                )
                valid_counts.append(int(valid.sum()))
                torso_counts.append(int(valid[TORSO_ANCHORS].sum()))
                diagonals.append(float(features.diagonal))
                candidate, _ = candidate_from_person(person, raw_index, threshold, quality)
                if candidate is not None:
                    accepted += 1
                    accepted_in_frame += 1
            zero_accepted_frames += accepted_in_frame == 0

    def quantiles(values: list[float]) -> dict[str, float]:
        return {
            key: float(value)
            for key, value in zip(
                ("p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99"),
                np.quantile(values, (.01, .05, .10, .25, .50, .75, .90, .95, .99)),
                strict=True,
            )
        }

    result = {
        "purpose": "candidate-geometry calibration only; no labels or validation accuracy used",
        "sampling": "10 manifest-order-spaced clips per activity, 10 activities",
        "clips": len(sample),
        "frames": frames,
        "legacy_valid_candidates": raw_candidates,
        "v2_accepted_candidates": accepted,
        "candidate_retention_rate": accepted / raw_candidates,
        "frames_with_no_v2_candidate": zero_accepted_frames,
        "frames_with_no_v2_candidate_rate": zero_accepted_frames / frames,
        "selected_thresholds": {
            "confidence_threshold": threshold,
            "min_valid_joints": quality.min_valid_joints,
            "min_torso_anchors": quality.min_torso_anchors,
            "min_diagonal_px": quality.min_diagonal_px,
        },
        "observed_candidate_quantiles": {
            "valid_joints": quantiles(valid_counts),
            "torso_anchors": quantiles(torso_counts),
            "diagonal_px": quantiles(diagonals),
        },
    }
    output = ROOT / "artifacts" / "tracker_v2_audit" / "candidate_filter_calibration.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

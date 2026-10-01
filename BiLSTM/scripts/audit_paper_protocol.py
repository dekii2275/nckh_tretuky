#!/usr/bin/env python3
"""Audit published settings and count samples without changing data or splits."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import read_manifest
from src.data.splits import validate_no_leakage
from src.reproducibility import capture_provenance, sha256_file
from src.utils import cfg_path, json_dump, load_config


# These values are explicit in Section III.C-D / Table II of the supplied PDF.
# Unreported details deliberately do not appear in this list.
REPORTED = {
    "sequence.length": 60,
    "model.bilstm.input_size": 50,
    "model.bilstm.hidden_size": 128,
    "model.bilstm.num_layers": 2,
    "model.bilstm.bidirectional": True,
    "model.attention.embed_dim": 256,
    "model.attention.num_heads": 4,
    "model.residual": True,
    "model.layer_norm": True,
    "model.pooling.type": "temporal_mean",
    "model.projection.in_features": 256,
    "model.projection.out_features": 64,
    "model.projection.activation": "relu",
    "model.dropout": 0.3,
    "model.classifier.in_features": 64,
    "model.classifier.num_classes": 10,
    "split.strategy": "session_holdout",
    "split.train_fraction": 0.8,
    "split.val_fraction": 0.2,
    "training.epochs": 100,
    "training.batch_size": 32,
    "training.optimizer.type": "Adam",
    "training.optimizer.learning_rate": 0.001,
    "training.loss.type": "CrossEntropyLoss",
    "early_stopping.enabled": True,
    "early_stopping.patience": 10,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/paper_repro.yaml")
    parser.add_argument("--paper", type=Path, help="Optional supplied PDF to fingerprint")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    checks = []
    for field, expected in REPORTED.items():
        actual = config
        for key in field.split("."):
            actual = actual.get(key) if isinstance(actual, dict) else None
        checks.append({"field": field, "paper": expected, "local": actual, "matches": actual == expected})
    artifacts = cfg_path(config, "outputs", "artifacts_dir", default="artifacts")
    split_dir = cfg_path(config, "outputs", "split_dir", default="artifacts/splits")
    manifest = read_manifest(artifacts / "manifest.csv")
    splits = {name: read_manifest(split_dir / f"{name}.csv") for name in ("train", "val")}
    if (split_dir / "test.csv").exists():
        splits["test"] = read_manifest(split_dir / "test.csv")
    validate_no_leakage(splits, str(config.get("split", {}).get("strategy", "session_holdout")))
    for name, rows in {"manifest": manifest, **splits}.items():
        if not rows or any(int(row["num_frames"]) <= 0 for row in rows):
            raise ValueError(f"Empty/non-positive-length samples in {name}")
    # Check full row identity too: a changed label or length must not pass a
    # clip-ID-only inventory check.
    expected_rows = {row["clip_id"]: row for row in manifest}
    combined = [row for rows in splits.values() for row in rows]
    if len(expected_rows) != len(manifest) or len(combined) != len(manifest):
        raise ValueError("Manifest and split sizes/unique clip IDs disagree")
    if any(expected_rows.get(row["clip_id"]) != row for row in combined):
        raise ValueError("Split rows differ from the manifest")
    counts = {}
    for name, rows in splits.items():
        lengths = [int(row["num_frames"]) for row in rows]
        counts[name] = {
            "clips": len(rows), "fraction": len(rows) / len(manifest),
            "frames": sum(lengths), "median_clip_frames": statistics.median(lengths),
            "min_clip_frames": min(lengths), "max_clip_frames": max(lengths),
            "clips_shorter_than_60": sum(length < 60 for length in lengths),
            "subject_session_groups": len({row["session_group"] for row in rows}),
            "classes": dict(sorted(Counter(row["activity"] for row in rows).items())),
            "hypothetical_full_window_counts": {
                str(stride): sum(max(0, (length - 60) // stride + 1) for length in lengths)
                for stride in (1, 30, 60)
            },
        }
    report = {
        "paper_doi": "10.1109/ICCA66035.2025.11431081",
        "paper_sha256": sha256_file(args.paper) if args.paper else None,
        "reported_config_checks": checks,
        "reported_config_checks_pass": all(row["matches"] for row in checks),
        "exact_author_reproduction_verified": False,
        "unresolved_author_details": [
            "Final clip inventory and exclusions for the ten-class experiment",
            "60-frame sample construction, stride, tail/short-clip policy and evaluation unit",
            "Session key definition, exact train/validation memberships and seed",
            "Person selection/tracking, missing/confidence keypoints and normalization edge cases",
            "Early-stopping monitor, internal dropout and remaining optimizer/framework settings",
        ],
        "manifest_clips": len(manifest),
        "split_integrity": "passed; complete clip partition with configured group separation",
        "split_counts": counts,
        "window_count_note": "Hypotheses only: full consecutive 60-frame windows, no short-clip padding or tail window; no samples were created and no author stride is inferred.",
        "provenance": capture_provenance(config),
    }
    output = args.output or artifacts / "protocol_audit.json"
    json_dump(report, output)
    print(json.dumps({key: value for key, value in report.items() if key not in {"provenance", "reported_config_checks"}}, indent=2))
    print(f"Saved {output}")
    if not report["reported_config_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

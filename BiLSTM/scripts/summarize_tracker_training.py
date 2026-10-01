#!/usr/bin/env python3
"""Create a compact, reproducible V2-minus-V1 training comparison."""

from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import sys


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "tracker_training"
sys.path.insert(0, str(ROOT))

from src.utils import load_config


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def best_epoch_record(run: str, checkpoint_epoch: int) -> dict:
    history = load_json(ROOT / "runs" / run / "history.json")["history"]
    return next(row for row in history if row["epoch"] == checkpoint_epoch)


def load_class_rows(version: str) -> list[dict]:
    with (OUT / version / "per_class_metrics.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_matrix(version: str) -> list[list[int]]:
    with (OUT / version / "confusion_matrix_counts.csv").open(newline="", encoding="utf-8") as handle:
        return [[int(value) for value in row] for row in csv.reader(handle)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    results = {version: load_json(OUT / version / "validation_results.json") for version in ("v1", "v2")}
    rows = {
        "v1": best_epoch_record("tracker_v1_controlled", results["v1"]["checkpoint_epoch"]),
        "v2": best_epoch_record("tracker_v2_controlled", results["v2"]["checkpoint_epoch"]),
    }
    scalar_names = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "loss")
    configs = {
        "v1": load_config(ROOT / "configs" / "paper_repro_tracker_v1.yaml"),
        "v2": load_config(ROOT / "configs" / "paper_repro_tracker_v2.yaml"),
    }
    controlled = {}
    for version, config in configs.items():
        clean = deepcopy({key: value for key, value in config.items() if not key.startswith("_")})
        clean.pop("tracking", None)
        outputs = clean.pop("outputs")
        controlled[version] = {"settings": clean, "outputs": outputs}
    manifest_paths = {
        "manifest.csv": ROOT / "artifacts" / "paper_repro" / "manifest.csv",
        "splits/train.csv": ROOT / "artifacts" / "splits" / "paper_repro" / "train.csv",
        "splits/val.csv": ROOT / "artifacts" / "splits" / "paper_repro" / "val.csv",
    }
    summary = {
        "comparison": "controlled V2 - V1; only tracking.mode/config differs",
        "seed": 42,
        "evaluation_split": "val",
        "evaluation_size": results["v1"]["evaluation_size"],
        "control_verification": {
            "non_tracking_non_output_settings_identical": controlled["v1"]["settings"] == controlled["v2"]["settings"],
            "shared_artifacts_dir": controlled["v1"]["outputs"]["artifacts_dir"] == controlled["v2"]["outputs"]["artifacts_dir"],
            "shared_split_dir": controlled["v1"]["outputs"]["split_dir"] == controlled["v2"]["outputs"]["split_dir"],
            "manifest_sha256": {name: sha256(path) for name, path in manifest_paths.items()},
            "separate_checkpoint_and_run_directories": (
                controlled["v1"]["outputs"]["checkpoints_dir"] != controlled["v2"]["outputs"]["checkpoints_dir"]
                and controlled["v1"]["outputs"]["runs_dir"] != controlled["v2"]["outputs"]["runs_dir"]
            ),
        },
        "v1": {
            "tracker": "legacy",
            "best_epoch": results["v1"]["checkpoint_epoch"],
            "train_accuracy_at_best_epoch": rows["v1"]["train_accuracy"],
            "train_macro_f1_at_best_epoch": rows["v1"]["train_macro_f1"],
            **{name: results["v1"][name] for name in scalar_names},
            "checkpoint_sha256": results["v1"]["checkpoint_sha256"],
        },
        "v2": {
            "tracker": "tracklet_v2",
            "best_epoch": results["v2"]["checkpoint_epoch"],
            "train_accuracy_at_best_epoch": rows["v2"]["train_accuracy"],
            "train_macro_f1_at_best_epoch": rows["v2"]["train_macro_f1"],
            **{name: results["v2"][name] for name in scalar_names},
            "checkpoint_sha256": results["v2"]["checkpoint_sha256"],
        },
    }
    summary["delta_v2_minus_v1"] = {
        "best_epoch": summary["v2"]["best_epoch"] - summary["v1"]["best_epoch"],
        "train_accuracy_at_best_epoch": (
            summary["v2"]["train_accuracy_at_best_epoch"] - summary["v1"]["train_accuracy_at_best_epoch"]
        ),
        "train_macro_f1_at_best_epoch": (
            summary["v2"]["train_macro_f1_at_best_epoch"] - summary["v1"]["train_macro_f1_at_best_epoch"]
        ),
        **{name: summary["v2"][name] - summary["v1"][name] for name in scalar_names},
    }
    if not all(
        summary["control_verification"][key]
        for key in (
            "non_tracking_non_output_settings_identical",
            "shared_artifacts_dir",
            "shared_split_dir",
            "separate_checkpoint_and_run_directories",
        )
    ):
        raise RuntimeError("V1/V2 experiment controls are not identical")
    with (OUT / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    v1_classes, v2_classes = load_class_rows("v1"), load_class_rows("v2")
    with (OUT / "per_class_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "activity_id", "activity", "support", "v1_recall", "v2_recall", "delta_recall",
            "v1_f1", "v2_f1", "delta_f1",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for old, new in zip(v1_classes, v2_classes, strict=True):
            if (old["activity_id"], old["activity"], old["support"]) != (
                new["activity_id"], new["activity"], new["support"]
            ):
                raise ValueError("Per-class rows are not aligned")
            writer.writerow({
                "activity_id": old["activity_id"],
                "activity": old["activity"],
                "support": old["support"],
                "v1_recall": old["recall"],
                "v2_recall": new["recall"],
                "delta_recall": float(new["recall"]) - float(old["recall"]),
                "v1_f1": old["f1"],
                "v2_f1": new["f1"],
                "delta_f1": float(new["f1"]) - float(old["f1"]),
            })

    names = [row["activity"] for row in v1_classes]
    v1_matrix, v2_matrix = load_matrix("v1"), load_matrix("v2")
    with (OUT / "confusion_matrix_delta_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *names])
        for name, old, new in zip(names, v1_matrix, v2_matrix, strict=True):
            writer.writerow([name, *(b - a for a, b in zip(old, new, strict=True))])


if __name__ == "__main__":
    main()

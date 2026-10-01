"""Protocol guards and clip-level evaluation for Task 3 sequence ablations."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .data.dataset import read_manifest
from .data.sequence import SequenceBuilder, canonical_mode
from .data.splits import validate_no_leakage
from .metrics import classification_metrics
from .reproducibility import sha256_file
from .utils import cfg_path, ensure_dir, load_config

ROOT = Path(__file__).resolve().parents[1]
TASK3_MODES = {"full_clip_resample", "single_window", "window_60_stride_60", "window_60_stride_30"}


def is_task3_config(config: dict[str, Any]) -> bool:
    return str(config.get("experiment", "")).startswith("TASK3_")


def validate_task3_config(config: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    if not is_task3_config(config):
        raise ValueError("Not a TASK3 sequence experiment")
    mode = canonical_mode(dict(config.get("sequence", {})))
    if mode not in TASK3_MODES:
        raise ValueError(f"Task 3 does not allow mode {mode}")
    if int(config["sequence"].get("length", config["sequence"].get("window_length", 60))) != 60:
        raise ValueError("Task 3 freezes sequence length at 60")
    if str(config["sequence"].get("short_clip_policy", "interpolate")) != "interpolate":
        raise ValueError("Task 3 must retain short clips with deterministic interpolation")

    frozen = load_config(ROOT / "configs/paper_repro_tracker_v2.yaml")
    protected = ("seed", "data", "tracking", "classes", "split", "model", "early_stopping", "evaluation")
    changed = [key for key in protected if config.get(key) != frozen.get(key)]
    base_training = dict(frozen["training"])
    actual_training = dict(config["training"])
    sampler = actual_training.pop("sampler", "all_sequences")
    if actual_training != base_training or sampler not in {"all_sequences", "clip_balanced"}:
        changed.append("training")
    if changed:
        raise ValueError("Frozen Tracker-V2 settings changed: " + ", ".join(changed))
    if mode in {"full_clip_resample", "single_window"} and sampler != "all_sequences":
        raise ValueError("One-sequence modes must use all_sequences")

    split_dir = cfg_path(config, "outputs", "split_dir", default="")
    expected_split = (ROOT / "artifacts/splits/paper_repro").resolve()
    if split_dir != expected_split:
        raise ValueError("Task 3 must reuse the frozen paper_repro split")
    splits = {name: read_manifest(split_dir / f"{name}.csv") for name in ("train", "val")}
    validate_no_leakage(splits, "session_holdout")
    if [len(splits[name]) for name in ("train", "val")] != [1091, 283]:
        raise ValueError("Frozen Task-3 clip counts must be 1091/283")
    train_ids = {row["clip_id"] for row in splits["train"]}
    val_ids = {row["clip_id"] for row in splits["val"]}
    train_groups = {f"{row['subject_id']}::{row['session_id']}" for row in splits["train"]}
    val_groups = {f"{row['subject_id']}::{row['session_id']}" for row in splits["val"]}
    train_paths = {str(Path(row["path"]).resolve()) for row in splits["train"]}
    val_paths = {str(Path(row["path"]).resolve()) for row in splits["val"]}
    if train_ids & val_ids or train_groups & val_groups or train_paths & val_paths:
        raise RuntimeError("Task-3 pre-expansion leakage guard failed")
    return splits


def protected_input_hashes() -> dict[str, str]:
    paths = [
        ROOT / "artifacts/paper_repro/manifest.csv",
        ROOT / "artifacts/splits/paper_repro/train.csv",
        ROOT / "artifacts/splits/paper_repro/val.csv",
        ROOT / "configs/paper_repro_tracker_v2.yaml",
        ROOT / "src/data/tracking.py",
        ROOT / "src/data/tracklet_tracking.py",
        ROOT / "src/data/preprocessing.py",
        ROOT / "src/models/bilstm_attention.py",
        ROOT / "checkpoints/tracker_v2_controlled/best.pt",
    ]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def aggregate_clip_logits(logits, metadata, truth):
    grouped: dict[str, dict[str, Any]] = {}
    for values, row, label in zip(logits, metadata, truth, strict=True):
        key = str(row.get("source_clip_id", row["clip_id"]))
        if key not in grouped:
            grouped[key] = {"metadata": row, "label": int(label), "logits": []}
        item = grouped[key]
        if item["label"] != int(label):
            raise ValueError(f"Inconsistent labels for source clip {key}")
        item["logits"].append(values)
    labels, predictions, rows = [], [], []
    for clip_id, item in grouped.items():
        mean_logits = np.asarray(item["logits"], dtype=np.float64).mean(axis=0)
        prediction = int(mean_logits.argmax())
        meta = item["metadata"]
        labels.append(item["label"])
        predictions.append(prediction)
        rows.append({
            "clip_id": clip_id,
            "subject_id": meta["subject_id"],
            "session_id": meta["session_id"],
            "activity": meta.get("activity"),
            "true_id": item["label"],
            "predicted_id": prediction,
            "num_sequences": len(item["logits"]),
            **{f"mean_logit_{i}": float(value) for i, value in enumerate(mean_logits)},
        })
    return labels, predictions, rows


def clip_metrics_from_stats(stats: dict[str, Any], class_names: list[str]):
    truth, predictions, rows = aggregate_clip_logits(stats["logits"], stats["metadata"], stats["y_true"])
    overall, per_class, matrix = classification_metrics(truth, predictions, len(class_names), class_names)
    return overall, per_class, matrix, rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_task3_evaluation(output: Path, stats: dict[str, Any], class_names: list[str]) -> dict[str, Any]:
    ensure_dir(output)
    sequence_overall, sequence_per_class, sequence_matrix = classification_metrics(
        stats["y_true"], stats["y_pred"], len(class_names), class_names
    )
    clip_overall, clip_per_class, clip_matrix, clip_rows = clip_metrics_from_stats(stats, class_names)
    _write_csv(output / "sequence_per_class_metrics.csv", sequence_per_class)
    _write_csv(output / "clip_per_class_metrics.csv", clip_per_class)
    _write_csv(output / "clip_predictions.csv", clip_rows)
    sequence_rows = []
    for meta, truth, pred, logits in zip(stats["metadata"], stats["y_true"], stats["y_pred"], stats["logits"], strict=True):
        sequence_rows.append({**meta, "true_id": truth, "predicted_id": pred,
                              **{f"logit_{i}": value for i, value in enumerate(logits)}})
    _write_csv(output / "sequence_predictions.csv", sequence_rows)
    for level, matrix in (("sequence", sequence_matrix), ("clip", clip_matrix)):
        np.savetxt(output / f"{level}_confusion_matrix_counts.csv", matrix, delimiter=",", fmt="%d")
        support = matrix.sum(axis=1, keepdims=True)
        percent = np.divide(100.0 * matrix, support, out=np.zeros_like(matrix, dtype=float), where=support != 0)
        np.savetxt(output / f"{level}_confusion_matrix_percent.csv", percent, delimiter=",", fmt="%.8f")
    return {
        **{f"sequence_{key}": value for key, value in sequence_overall.items()},
        **{f"clip_{key}": value for key, value in clip_overall.items()},
        "sequence_evaluation_size": len(stats["y_true"]),
        "clip_evaluation_size": len(clip_rows),
        "clip_aggregation": "arithmetic mean of pre-softmax logits, then argmax",
    }


def sequence_count_audit(config: dict[str, Any], splits: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    builder = SequenceBuilder(dict(config["sequence"]))
    result: dict[str, Any] = {"mode": builder.mode}
    for split, records in splits.items():
        samples = builder.expand(records)
        per_clip: dict[str, int] = defaultdict(int)
        per_class: dict[str, int] = defaultdict(int)
        per_subject: dict[str, int] = defaultdict(int)
        per_session: dict[str, int] = defaultdict(int)
        for sample in samples:
            record = records[int(sample["clip_index"])]
            per_clip[str(record["clip_id"])] += 1
            per_class[str(record["activity"])] += 1
            per_subject[str(record["subject_id"])] += 1
            per_session[f"{record['subject_id']}::{record['session_id']}"] += 1
        values = [per_clip[str(record["clip_id"])] for record in records]
        result[split] = {
            "clips": len(records), "sequences": len(samples),
            "short_clips_interpolated": sum(int(record["num_frames"]) < 60 for record in records),
            "windows_per_clip": {"min": min(values), "median": float(np.median(values)),
                                 "mean": float(np.mean(values)), "max": max(values)},
            "sequences_per_class": dict(sorted(per_class.items())),
            "sequences_per_subject": dict(sorted(per_subject.items())),
            "sequences_per_session": dict(sorted(per_session.items())),
        }
    return result


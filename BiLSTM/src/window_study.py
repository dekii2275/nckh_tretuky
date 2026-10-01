"""Frozen protocol guards, sample audits and two-level evaluation."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .data.dataset import read_manifest, window_metadata
from .data.splits import validate_no_leakage
from .metrics import classification_metrics
from .reproducibility import sha256_file
from .utils import cfg_path, json_dump, load_config

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {60: "S1_window60_stride60", 30: "S2_window60_stride30", 15: "S3_window60_stride15"}


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parameter_breakdown(model):
    modules = {"BiLSTM": model.bilstm, "MHA": model.attention, "LayerNorm": model.layer_norm,
               "Projection": model.projection, "Classifier": model.classifier}
    result = {name: sum(p.numel() for p in module.parameters() if p.requires_grad)
              for name, module in modules.items()}
    result["Total"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(json.dumps({"parameter_breakdown": result}), flush=True)
    if result["Total"] != 860362 or sum(v for k, v in result.items() if k != "Total") != 860362:
        raise RuntimeError("Frozen paper model must have exactly 860362 parameters")
    return result


def protected_hashes():
    paths = [ROOT / "configs/paper_repro.yaml", ROOT / "src/models/bilstm_attention.py",
             ROOT / "src/data/child_tracking.py", ROOT / "src/data/preprocessing.py",
             ROOT / "src/data/openpose_parser.py", ROOT / "src/seed.py"]
    for folder in ("artifacts/paper_repro", "artifacts/splits/paper_repro", "runs/paper_repro", "checkpoints/paper_repro"):
        paths.extend(p for p in (ROOT / folder).rglob("*") if p.is_file())
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(paths)}


def verify_preserved():
    path = ROOT / "artifacts/window_audit/protected_hashes.json"
    current = protected_hashes()
    if path.exists():
        if json.loads(path.read_text()) != current:
            raise RuntimeError("Protected R0, split, or frozen implementation changed")
    else:
        json_dump(current, path)
    return current


def frozen_inputs(config):
    base = load_config(ROOT / "configs/paper_repro.yaml")
    allowed = {"sequence", "outputs", "experiment"}
    for key in set(base) | set(config):
        if not key.startswith("_") and key not in allowed and base.get(key) != config.get(key):
            raise ValueError(f"Frozen setting changed: {key}")
    split_dir = cfg_path(config, "outputs", "split_dir", default="")
    if split_dir != (ROOT / "artifacts/splits/paper_repro").resolve():
        raise ValueError("Window experiments must reuse the frozen paper_repro split")
    seq = config["sequence"]
    stride = int(seq["stride"])
    if seq.get("mode") != "sliding_window" or stride not in EXPERIMENTS:
        raise ValueError("Only the three predeclared sliding-window protocols are allowed")
    if config.get("experiment") != EXPERIMENTS[stride]:
        raise ValueError("Experiment name/stride mismatch")
    for key, folder in (("artifacts_dir", "artifacts"), ("runs_dir", "runs"), ("checkpoints_dir", "checkpoints")):
        if cfg_path(config, "outputs", key, default="") != ROOT / folder / f"window_s{stride}":
            raise ValueError(f"Unsafe experiment output directory: {key}")
    verify_preserved()
    splits = {name: read_manifest(split_dir / f"{name}.csv") for name in ("train", "val")}
    validate_no_leakage(splits, "session_holdout")
    canonical = [{str(Path(r["path"]).resolve()) for r in splits[name]} for name in ("train", "val")]
    if canonical[0] & canonical[1]:
        raise RuntimeError("Canonical source paths overlap")
    if [len(splits[k]) for k in ("train", "val")] != [1091, 283]:
        raise ValueError("Frozen clip counts differ from R0")
    manifest = read_manifest(ROOT / "artifacts/paper_repro/manifest.csv")
    if {r["clip_id"]: r for r in manifest} != {r["clip_id"]: r for rows in splits.values() for r in rows}:
        raise ValueError("Frozen manifest differs from the split union")
    return splits


def audit_windows():
    verify_preserved()
    output = ROOT / "artifacts/window_audit"
    counts, summaries = [], {}
    for stride, experiment in EXPERIMENTS.items():
        config = load_config(ROOT / f"configs/window_s{stride}.yaml")
        splits = frozen_inputs(config)  # Validate BEFORE expansion.
        summary = {"experiment": experiment, "stride": stride, "window_length": 60,
                   "leakage": {"clip_id_intersection": 0, "path_intersection": 0, "subject_session_intersection": 0}}
        for split, records in splits.items():
            windows = window_metadata(records, config["sequence"])
            per_clip = Counter(w["clip_id"] for w in windows)
            per_class = Counter(w["activity"] for w in windows)
            per_group = Counter(f"{w['subject_id']}::{w['session_id']}" for w in windows)
            values = [per_clip[r["clip_id"]] for r in records]
            dropped = Counter(r["activity"] for r in records if not per_clip[r["clip_id"]])
            tails = []
            for row, n in zip(records, values, strict=True):
                t = int(row["num_frames"])
                tail = t - ((n - 1) * stride + 60) if n else 0
                tails.append(tail)
                counts.append({"experiment": experiment, "split": split, "clip_id": row["clip_id"],
                               "subject_id": row["subject_id"], "session_id": row["session_id"],
                               "activity": row["activity"], "activity_id": row["activity_id"],
                               "num_source_frames": t, "windows": n, "dropped_short_clip": int(n == 0),
                               "dropped_short_frames": t if n == 0 else 0, "dropped_tail_frames": tail})
            summary[split] = {
                "clips": len(records), "retained_clips": sum(n > 0 for n in values), "windows": len(windows),
                "clips_T_lt_60": sum(int(r["num_frames"]) < 60 for r in records),
                "dropped_clips": sum(n == 0 for n in values),
                "dropped_clips_per_class": {r["activity"]: dropped[r["activity"]] for r in records},
                "windows_per_class": dict(sorted(per_class.items())),
                "windows_per_subject_session": {f"{r['subject_id']}::{r['session_id']}": per_group[f"{r['subject_id']}::{r['session_id']}"] for r in records},
                "mean_windows_per_clip": float(np.mean(values)), "median_windows_per_clip": float(np.median(values)),
                "min_windows_per_clip": min(values), "max_windows_per_clip": max(values),
                "window_statistics_population": "all frozen clips, including dropped clips as zero",
                "dropped_tail_frames": sum(tails), "clips_with_dropped_tail": sum(t > 0 for t in tails),
            }
            write_csv(output / f"window_s{stride}_{split}_metadata.csv",
                      [{k: v for k, v in w.items() if k != "clip_index"} for w in windows])
        summaries[experiment] = summary
    write_csv(output / "window_counts.csv", counts)
    json_dump({"R0": {"train_clips": 1091, "val_clips": 283, "samples_per_clip": 1},
               "experiments": summaries, "note": "Predeclared sensitivity protocols; author stride is unknown."},
              output / "window_summary.json")
    return summaries


def aggregate_clip_logits(logits, metadata, truth):
    grouped = {}
    for values, row, label in zip(logits, metadata, truth, strict=True):
        key = row["clip_id"]
        if key not in grouped:
            grouped[key] = {"metadata": row, "label": int(label), "logits": []}
        item = grouped[key]
        if item["label"] != int(label) or item["metadata"]["path"] != row["path"]:
            raise ValueError(f"Inconsistent clip identity: {key}")
        item["logits"].append(values)
    rows, labels, predictions = [], [], []
    for clip_id, item in grouped.items():
        mean = np.asarray(item["logits"], dtype=np.float64).mean(axis=0)
        prediction = int(mean.argmax())
        labels.append(item["label"])
        predictions.append(prediction)
        rows.append({"clip_id": clip_id, "subject_id": item["metadata"]["subject_id"],
                     "session_id": item["metadata"]["session_id"], "source_path": item["metadata"]["path"],
                     "true_id": item["label"], "predicted_id": prediction, "num_windows": len(item["logits"]),
                     **{f"mean_logit_{i}": float(v) for i, v in enumerate(mean)}})
    return labels, predictions, rows


def save_level(output, level, truth, predictions, names):
    overall, per_class, matrix = classification_metrics(truth, predictions, len(names), names)
    support = matrix.sum(axis=1, keepdims=True)
    percent = np.divide(100.0 * matrix, support, out=np.zeros_like(matrix, dtype=float), where=support != 0)
    np.savetxt(output / f"{level}_confusion_matrix_counts.csv", matrix, delimiter=",", fmt="%d")
    np.savetxt(output / f"{level}_confusion_matrix_percent.csv", percent, delimiter=",", fmt="%.10f")
    write_csv(output / f"{level}_per_class_metrics.csv", per_class)
    return {**{f"{level}_{k}": v for k, v in overall.items()}, f"{level}_evaluation_size": len(truth)}


def save_window_evaluation(output, stats, names):
    output.mkdir(parents=True, exist_ok=True)
    result = save_level(output, "window", stats["y_true"], stats["y_pred"], names)
    truth, predicted, clips = aggregate_clip_logits(stats["logits"], stats["metadata"], stats["y_true"])
    result.update(save_level(output, "clip", truth, predicted, names))
    result["clip_aggregation"] = "mean_logits"
    result["clip_population"] = "validation clips with at least one full 60-frame window"
    write_csv(output / "clip_predictions.csv", clips)
    windows = [{**row, "true_id": t, "predicted_id": p, **{f"logit_{i}": v for i, v in enumerate(logit)}}
               for row, t, p, logit in zip(stats["metadata"], stats["y_true"], stats["y_pred"], stats["logits"], strict=True)]
    write_csv(output / "window_predictions.csv", windows)
    return result

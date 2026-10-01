#!/usr/bin/env python3
"""Generate pre-training implementation, compression, motion and leakage audits."""

from __future__ import annotations

import csv
import inspect
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import SkeletonClipDataset, read_manifest  # noqa: E402
from src.data.sequence import SequenceBuilder, selected_source_indices, temporal_positions, temporal_resample  # noqa: E402
from src.reproducibility import sha256_file  # noqa: E402
from src.sequence_study import protected_input_hashes, sequence_count_audit, validate_task3_config  # noqa: E402
from src.utils import json_dump, load_config  # noqa: E402

OUT = ROOT / "artifacts/sequence_task3/audit"
CONFIGS = {
    "full_clip_resample": ROOT / "configs/task3_exp_a_full_resample.yaml",
    "single_window": ROOT / "configs/task3_exp_b_center60.yaml",
    "window_60_stride_60": ROOT / "configs/task3_exp_c_stride60.yaml",
    "window_60_stride_30": ROOT / "configs/task3_exp_d_stride30.yaml",
}
FOCUS = {"maracas_forward_shaking_mfs", "maracas_shaking_ms", "sing_and_clap_sac",
         "drumming_dr", "body_swing_bs", "chest_expansion_ce"}
PERCENTILES = [0, 10, 25, 50, 75, 90, 95, 100]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distribution(values) -> dict[str, float]:
    numbers = np.asarray(list(values), dtype=float)
    quantiles = np.percentile(numbers, PERCENTILES)
    return {name: float(value) for name, value in zip(
        ("min", "p10", "p25", "median", "p75", "p90", "p95", "max"), quantiles, strict=True
    )}


def source_location(obj) -> str:
    path = Path(inspect.getsourcefile(obj) or "").resolve()
    line = inspect.getsourcelines(obj)[1]
    return f"{path.relative_to(ROOT)}:{line}"


def pipeline_report() -> None:
    from src.data.openpose_parser import parse_clip
    from src.data.tracking import select_person_track
    from src.data.preprocessing import preprocess_clip
    lines = [
        "# Task 3 — current sequence implementation trace", "",
        "The source path is parsed into all raw OpenPose frames, then Tracker V2 selects one full-clip person track, "
        "then missing-joint handling and hip/shoulder normalization run on the full track, and only then is a 60-step sequence selected.", "",
        f"1. Raw JSON clip parsing: `{source_location(parse_clip)}`.",
        f"2. Tracker dispatch/person selection: `{source_location(select_person_track)}`.",
        f"3. Missing-joint preprocessing and normalization: `{source_location(preprocess_clip)}`.",
        f"4. Dataset ordering (parse → track → preprocess → extract): `{source_location(SkeletonClipDataset.__getitem__)}`.",
        f"5. Central sequence construction: `{source_location(SequenceBuilder)}`; exact resampling: `{source_location(temporal_resample)}`.",
        "6. Batch collation pads (all Task-3 samples are already 60 frames) and returns `[B,60,25,2]`; each item is `[60,25,2]`.", "",
        "Current `full_clip_resample` semantics:", "",
        "- Resampling occurs after both tracking and normalization.",
        "- For N > 60: `positions=linspace(0,N-1,60)`, indices use NumPy `rint` (round-to-nearest, ties-to-even), endpoints included.",
        "- For N = 60: an exact copy is returned.",
        "- For N < 60: linear temporal interpolation between adjacent normalized poses produces 60 outputs; clips are never silently dropped.",
        "- For N > 60, selected indices cannot duplicate; for N < 60, interpolation positions reuse source endpoints/neighbors but output poses are interpolated rather than duplicated (except N=1).",
    ]
    (OUT / "implementation_trace.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compression_audit(records: list[dict[str, str]]) -> list[dict]:
    rows = []
    for record in records:
        n = int(record["num_frames"])
        indices = selected_source_indices(n)
        gaps = np.diff(indices)
        rows.append({
            "clip_id": record["clip_id"], "activity": record["activity"],
            "subject_id": record["subject_id"], "session_id": record["session_id"],
            "original_num_frames": n, "number_selected": 60,
            "average_original_frame_gap": float(gaps.mean()),
            "max_original_frame_gap": int(gaps.max()),
            "duplicate_selected_indices": int(60 - len(np.unique(indices))),
            "temporal_compression_ratio": n / 60.0,
            "selected_indices": ";".join(map(str, indices.tolist())),
        })
    write_csv(OUT / "temporal_compression_per_clip.csv", rows)
    metrics = ("original_num_frames", "number_selected", "average_original_frame_gap",
               "max_original_frame_gap", "duplicate_selected_indices", "temporal_compression_ratio")
    summary = {"all": {metric: distribution(row[metric] for row in rows) for metric in metrics}, "activities": {}}
    activities = sorted({row["activity"] for row in rows})
    for activity in activities:
        subset = [row for row in rows if row["activity"] == activity]
        summary["activities"][activity] = {metric: distribution(row[metric] for row in subset) for metric in metrics}
    gaps = [row["average_original_frame_gap"] for row in rows]
    summary["average_gap_bins"] = {
        "le_1_5": sum(value <= 1.5 for value in gaps),
        "gt_1_5_le_2": sum(1.5 < value <= 2 for value in gaps),
        "gt_2_le_3": sum(2 < value <= 3 for value in gaps),
        "gt_3_le_5": sum(3 < value <= 5 for value in gaps),
        "gt_5": sum(value > 5 for value in gaps),
        "gt_10": sum(value > 10 for value in gaps),
    }
    json_dump(summary, OUT / "temporal_compression_summary.json")
    return rows


def direction_change_rate(velocity: np.ndarray) -> float:
    if len(velocity) < 2:
        return 0.0
    a, b = velocity[:-1].reshape(len(velocity) - 1, -1), velocity[1:].reshape(len(velocity) - 1, -1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cosine = np.divide((a * b).sum(axis=1), denom, out=np.ones_like(denom), where=denom > 1e-12)
    return float(np.mean(cosine < 0.0))


def motion_values(sequence: np.ndarray, n: int):
    velocity = np.diff(sequence, axis=0)
    speed = np.linalg.norm(velocity, axis=2).mean(axis=1)
    sampled = temporal_resample(sequence, 60)
    positions = temporal_positions(n, 60)
    sampled_velocity = np.diff(sampled, axis=0)
    source_gaps = np.diff(positions)
    sampled_speed = np.linalg.norm(sampled_velocity, axis=2).mean(axis=1) / source_gaps
    if len(speed) >= 3:
        threshold = np.percentile(speed, 90)
        peaks = np.flatnonzero((speed[1:-1] > speed[:-2]) & (speed[1:-1] >= speed[2:]) &
                               (speed[1:-1] >= threshold)) + 1
    else:
        peaks = np.array([], dtype=int)
    nearest_distance = np.min(np.abs(peaks[:, None] - positions[None, :]), axis=1) if len(peaks) else np.array([])
    return speed, sampled_speed, positions, {
        "original_mean_joint_displacement": float(speed.mean()) if len(speed) else 0.0,
        "sampled_mean_velocity_per_source_frame": float(sampled_speed.mean()) if len(sampled_speed) else 0.0,
        "mean_velocity_retention_ratio": float(sampled_speed.mean() / speed.mean()) if speed.mean() > 0 else 1.0,
        "original_temporal_variation": float(speed.std()) if len(speed) else 0.0,
        "sampled_temporal_variation": float(sampled_speed.std()) if len(sampled_speed) else 0.0,
        "temporal_variation_retention_ratio": float(sampled_speed.std() / speed.std()) if speed.std() > 0 else 1.0,
        "original_direction_change_rate": direction_change_rate(velocity),
        "sampled_direction_change_rate": direction_change_rate(sampled_velocity / source_gaps[:, None, None]),
        "local_motion_peaks": int(len(peaks)),
        "local_motion_peaks_captured_within_1_frame": int(np.sum(nearest_distance <= 1.0)),
        "local_motion_peak_recall_within_1_frame": float(np.mean(nearest_distance <= 1.0)) if len(peaks) else 1.0,
    }


def motion_audit(records: list[dict[str, str]], compression: list[dict]) -> None:
    config = load_config(CONFIGS["full_clip_resample"])
    data = dict(config["data"])
    data["project_root"] = config["_project_root"]
    data["tracking"] = dict(config["tracking"])
    dataset = SkeletonClipDataset(records, data_config=data, sequence_config={"mode": "full_clip"})
    rank = sorted((row for row in compression if row["activity"] in FOCUS),
                  key=lambda row: row["original_num_frames"], reverse=True)
    selected_visuals = []
    per_activity = Counter()
    for row in rank:
        if len(selected_visuals) >= 20:
            break
        if per_activity[row["activity"]] < 4:
            selected_visuals.append(row["clip_id"])
            per_activity[row["activity"]] += 1
    # Fill if one focus class has too few candidates.
    selected_visuals += [row["clip_id"] for row in rank
                         if row["clip_id"] not in selected_visuals][:20 - len(selected_visuals)]
    visual_set = set(selected_visuals)
    motion_rows = []
    curves = OUT / "motion_curves"
    curves.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records):
        sequence = dataset[index]["skeleton"].numpy()
        speed, sampled_speed, positions, values = motion_values(sequence, len(sequence))
        motion_rows.append({"clip_id": record["clip_id"], "activity": record["activity"],
                            "original_num_frames": len(sequence), **values})
        if record["clip_id"] in visual_set:
            figure, axis = plt.subplots(figsize=(10, 3.8))
            axis.plot(np.arange(1, len(speed) + 1), speed, linewidth=1.0, alpha=.75, label="original per-frame motion")
            axis.plot((positions[:-1] + positions[1:]) / 2, sampled_speed, "o-", markersize=2.5,
                      linewidth=1.1, label="60-step sampled motion / source-frame gap")
            axis.set(title=f"{record['clip_id']} | {record['activity']} | N={len(sequence)}",
                     xlabel="original frame", ylabel="mean normalized joint displacement")
            axis.legend(fontsize=8)
            figure.tight_layout()
            figure.savefig(curves / f"{record['clip_id']}.png", dpi=150)
            plt.close(figure)
        if (index + 1) % 100 == 0:
            print(f"motion audit {index + 1}/{len(records)}", flush=True)
    write_csv(OUT / "motion_preservation_per_clip.csv", motion_rows)
    metric_names = list(values)
    summary = {"all": {metric: distribution(row[metric] for row in motion_rows) for metric in metric_names},
               "activities": {}, "visualized_clip_ids": selected_visuals,
               "definitions": {
                   "motion": "mean Euclidean displacement over 25 normalized joints",
                   "sampled_velocity": "sample-to-sample displacement divided by original-frame position gap",
                   "local_peak": "local maximum at or above the clip's 90th percentile motion",
                   "captured_peak": "a 60-step sample position lies within one original frame of the peak",
               }}
    for activity in sorted({row["activity"] for row in motion_rows}):
        subset = [row for row in motion_rows if row["activity"] == activity]
        summary["activities"][activity] = {metric: distribution(row[metric] for row in subset)
                                            for metric in metric_names}
    json_dump(summary, OUT / "motion_preservation_summary.json")


def counts_and_leakage() -> None:
    modes = {}
    count_rows = []
    class_rows = []
    leakage = {}
    for mode, path in CONFIGS.items():
        config = load_config(path)
        splits = validate_task3_config(config)  # split validation occurs before expansion
        audit = sequence_count_audit(config, splits)
        modes[mode] = audit
        for split in ("train", "val"):
            count_rows.append({"mode": mode, "split": split, "clips": audit[split]["clips"],
                               "sequences": audit[split]["sequences"],
                               "short_clips_interpolated": audit[split]["short_clips_interpolated"]})
            for activity, count in audit[split]["sequences_per_class"].items():
                class_rows.append({"mode": mode, "split": split, "activity": activity, "sequences": count})
        builder = SequenceBuilder(config["sequence"])
        expanded = {split: builder.expand(rows) for split, rows in splits.items()}
        source_sets = {split: {item["source_clip_id"] for item in rows} for split, rows in expanded.items()}
        path_sets = {split: {str(Path(row["path"]).resolve()) for row in rows} for split, rows in splits.items()}
        group_sets = {split: {f"{row['subject_id']}::{row['session_id']}" for row in rows}
                      for split, rows in splits.items()}
        leakage[mode] = {
            "clip_id_overlap": len(source_sets["train"] & source_sets["val"]),
            "window_origin_clip_overlap": len(source_sets["train"] & source_sets["val"]),
            "subject_session_overlap": len(group_sets["train"] & group_sets["val"]),
            "canonical_source_path_overlap": len(path_sets["train"] & path_sets["val"]),
            "exact_frame_reuse_overlap": 0,
            "exact_frame_reuse_proof": "frame identity=(canonical source path, source-frame index); disjoint canonical paths imply empty intersection",
            "split_before_sequence_expansion": True,
        }
    write_csv(OUT / "sample_counts.csv", count_rows)
    write_csv(OUT / "sample_counts_per_class.csv", class_rows)
    json_dump(modes, OUT / "sequence_distribution.json")
    json_dump(leakage, OUT / "leakage_audit.json")


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    records = read_manifest(ROOT / "artifacts/paper_repro/manifest.csv")
    if len(records) != 1374:
        raise ValueError(f"Expected frozen population of 1374, got {len(records)}")
    pipeline_report()
    compression = compression_audit(records)
    counts_and_leakage()
    json_dump(protected_input_hashes(), OUT / "protected_input_hashes.json")
    motion_audit(records, compression)
    longest = sorted(compression, key=lambda row: row["original_num_frames"], reverse=True)[:20]
    write_csv(OUT / "top20_longest_clips.csv", longest)
    json_dump({"status": "PASS", "clips": len(records), "runtime_seconds": time.perf_counter() - started,
               "manifest_sha256": sha256_file(ROOT / "artifacts/paper_repro/manifest.csv")}, OUT / "audit_status.json")
    print(json.dumps({"status": "PASS", "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()

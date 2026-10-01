#!/usr/bin/env python3
"""Measure unchanged full-clip preprocessing; associate saved S1 errors descriptively."""
import json
from collections import Counter, defaultdict

import numpy as np

from diagnostics_common import OUT, REVIEW, ROOT, json_dump, markdown_table, preserve_runs, read_labels, read_manifest, split_records, write_csv
from src.data.openpose_parser import parse_clip
from src.engine import make_dataset
from src.utils import load_config


def missing_runs(valid):
    """Return missing-joint run lengths and whether bounded/edge/never-observed."""
    for joint in range(valid.shape[1]):
        mask = ~valid[:, joint]
        edges = np.diff(np.r_[False, mask, False].astype(int))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
            kind = "never_observed_joint" if start == 0 and end == len(mask) else "leading_edge" if start == 0 else "trailing_edge" if end == len(mask) else "bounded_gap"
            yield kind, int(end - start)


def summarize(rows):
    frames = sum(r["num_frames"] for r in rows)
    joints = frames * 25
    result = {"clips": len(rows), "frames": frames, "joint_observations": joints}
    for key in ("invalid_midhip_frames", "missing_left_shoulder_frames", "missing_right_shoulder_frames",
                "invalid_shoulder_scale_frames", "torso_fallback_frames", "frames_gt25pct_joints_missing"):
        result[key] = sum(r[key] for r in rows)
        percent_key = "frames_gt25pct_joints_missing_percent" if key == "frames_gt25pct_joints_missing" else key.replace("_frames", "_percent")
        result[percent_key] = 100 * result[key] / frames if frames else None
    for key in ("missing_joints", "interpolated_joints", "bounded_interpolated_joints", "edge_filled_joints", "never_observed_joint_frames"):
        result[key] = sum(r[key] for r in rows)
        result[key + "_percent"] = 100 * result[key] / joints if joints else None
    result["severe_missingness_clips"] = sum(r["severe_missingness"] for r in rows)
    result["severe_missingness_clips_percent"] = 100 * result["severe_missingness_clips"] / len(rows) if rows else None
    return result


def error_factors(rows):
    predictions = {r["clip_id"]: r for r in read_manifest(ROOT / "artifacts/window_s60/clip_predictions.csv")}
    labels = {r["clip_id"]: r["review_label"] for r in read_labels()}
    errors = []
    for row in rows:
        if row["split"] != "val":
            continue
        prediction = predictions.get(row["clip_id"])
        if prediction:
            logits = np.asarray([float(prediction[f"mean_logit_{i}"]) for i in range(10)])
            probabilities = np.exp(logits - logits.max()); probabilities /= probabilities.sum()
            ranked = np.sort(logits)
            details = {"classification_correct": int(prediction["true_id"] == prediction["predicted_id"]),
                       "predicted_id": int(prediction["predicted_id"]), "prediction_confidence": float(probabilities.max()),
                       "prediction_logit_margin": float(ranked[-1] - ranked[-2]), "evaluation_status": "S1_retained"}
        else:
            if row["num_frames"] >= 60:
                raise ValueError(f"Missing S1 validation prediction: {row['clip_id']}")
            details = {"classification_correct": None, "predicted_id": None, "prediction_confidence": None,
                       "prediction_logit_margin": None, "evaluation_status": "dropped_short_clip"}
        errors.append({"clip_id": row["clip_id"], "subject_id": row["subject_id"], "session_id": row["session_id"],
                       "activity": row["activity"], "true_id": row["activity_id"], **details,
                       "missing_joint_rate": row["missing_joints"] / (25 * row["num_frames"]),
                       "torso_fallback_rate": row["torso_fallback_frames"] / row["num_frames"],
                       "mean_keypoint_confidence": row["mean_keypoint_confidence"],
                       "mean_valid_keypoint_confidence": row["mean_valid_keypoint_confidence"],
                       "tracking_review_label": labels.get(row["clip_id"], ""),
                       "num_people": row["mean_num_people"], "max_people": row["max_people"],
                       "raw_selected_index_change_rate": row["raw_selected_index_change_rate"],
                       "severe_missingness": row["severe_missingness"]})
    write_csv(OUT / "error_factor_analysis.csv", errors)
    summary = []
    fields = ("missing_joint_rate", "torso_fallback_rate", "mean_keypoint_confidence", "mean_valid_keypoint_confidence", "num_people", "prediction_confidence", "prediction_logit_margin", "raw_selected_index_change_rate")
    for activity in ["ALL"] + sorted({r["activity"] for r in errors}):
        for correct in (1, 0):
            group = [r for r in errors if r["classification_correct"] == correct and (activity == "ALL" or r["activity"] == activity)]
            summary.append({"activity": activity, "classification": "correct" if correct else "incorrect", "clips": len(group),
                            **{f"mean_{f}": float(np.mean([r[f] for r in group])) if group else None for f in fields},
                            "human_reviewed_clips": sum(bool(r["tracking_review_label"]) for r in group)})
    write_csv(OUT / "error_factor_comparisons.csv", summary)
    lines = ["# Descriptive S1 validation error factors", "", "Only the 275 retained validation clips contribute to correct/incorrect comparisons. Eight short clips remain in the per-clip CSV with blank prediction fields. Rates are fractions 0–1; confidence is the maximum softmax of mean logits, not calibrated probability.", "",
             "Keypoint confidence is averaged over all 25 selected raw joints per frame, including zero-confidence/missing joints and zero for no selected candidate. Mean valid-joint confidence is also reported. num_people is the time mean of raw OpenPose candidate counts, not distinct verified identities.", ""]
    lines += markdown_table(summary[:2], ["classification", "clips", "mean_missing_joint_rate", "mean_torso_fallback_rate", "mean_mean_keypoint_confidence", "mean_num_people", "mean_prediction_confidence", "human_reviewed_clips"])
    lines += ["", "These are unadjusted clip-weighted descriptions, with per-activity breakdown in the CSV. Differences can reflect class composition, duration, detection noise and other confounding. A changing raw selected index is not a verified identity switch. Without human labels, tracking ambiguity/error cannot be measured from it."]
    (OUT / "error_factor_analysis.md").write_text("\n".join(lines) + "\n")
    return summary[:2]


def main():
    preserve_runs()
    config = load_config(ROOT / "configs/paper_repro.yaml")
    records = [dict(r, split=s) for s, group in split_records().items() for r in group]
    # Same tracking/preprocessing implementation as R0, without temporal sample construction.
    dataset = make_dataset(records, {**config, "sequence": {"mode": "full_clip"}}, return_debug=True)
    gap_counts = Counter()
    results = []
    for i, record in enumerate(records):
        item = dataset[i]
        valid = item["valid_mask"].numpy()
        frames, _ = parse_clip(record["path"], strict=True)
        t = len(frames)
        if t != int(record["num_frames"]):
            raise ValueError(f"Raw/manifest frame count mismatch: {record['clip_id']}")
        selected = item["tracking"]["selected_person_per_frame"]
        raw = np.zeros((t, 25, 3), dtype=np.float32)
        for frame, index in enumerate(selected):
            if index >= 0:
                raw[frame] = frames[frame][index]
        widths = np.linalg.norm(raw[:, 5, :2] - raw[:, 2, :2], axis=1)
        shoulder_valid = valid[:, 5] & valid[:, 2] & np.isfinite(widths) & (widths > 5.0)
        local_gaps = Counter(missing_runs(valid))
        parts = {kind: sum(length * n for (k, length), n in local_gaps.items() if k == kind)
                 for kind in ("bounded_gap", "leading_edge", "trailing_edge", "never_observed_joint")}
        for (kind, length), n in local_gaps.items():
            gap_counts[(record["split"], record["activity"], kind, length)] += n
        metadata = item["preprocessing"]
        interpolated = parts["bounded_gap"] + parts["leading_edge"] + parts["trailing_edge"]
        assert np.isclose(100 * interpolated / (t * 25), metadata["interpolated_joint_percentage"])
        severe_frames = int(((~valid).mean(1) > 0.25).sum())
        row = {"clip_id": record["clip_id"], "subject_id": record["subject_id"], "session_id": record["session_id"],
               "activity": record["activity"], "activity_id": int(record["activity_id"]), "split": record["split"], "num_frames": t,
               "invalid_midhip_frames": int((~valid[:, 8]).sum()),
               "missing_left_shoulder_frames": int((~valid[:, 5]).sum()), "missing_right_shoulder_frames": int((~valid[:, 2]).sum()),
               "invalid_shoulder_scale_frames": int((~shoulder_valid).sum()),
               "torso_fallback_frames": int(metadata["scale_fallback_frames"]),
               "implementation_invalid_scale_counter": int(metadata["invalid_scale_frames"]),
               "missing_joints": int((~valid).sum()), "interpolated_joints": interpolated,
               "bounded_interpolated_joints": parts["bounded_gap"], "edge_filled_joints": parts["leading_edge"] + parts["trailing_edge"],
               "never_observed_joint_frames": parts["never_observed_joint"], "frames_gt25pct_joints_missing": severe_frames,
               "severe_missingness": int(severe_frames / t > 0.25),
               "mean_keypoint_confidence": float(np.where(np.isfinite(raw[:, :, 2]), raw[:, :, 2], 0).mean()),
               "mean_valid_keypoint_confidence": float(raw[:, :, 2][valid].mean()) if valid.any() else 0.0,
               "mean_num_people": float(np.mean([len(f) for f in frames])), "max_people": max(len(f) for f in frames),
               "raw_selected_index_change_rate": sum(a >= 0 and b >= 0 and a != b for a, b in zip(selected, selected[1:])) / max(t - 1, 1)}
        assert row["invalid_shoulder_scale_frames"] == row["torso_fallback_frames"]
        results.append(row)
        if (i + 1) % 100 == 0:
            print(f"Audited unchanged preprocessing {i+1}/{len(records)} clips", flush=True)
    write_csv(OUT / "preprocessing_clip_usage.csv", results)
    summary = []
    for split in ("train", "val"):
        for activity in ["ALL"] + sorted({r["activity"] for r in results}):
            group = [r for r in results if r["split"] == split and (activity == "ALL" or r["activity"] == activity)]
            summary.append({"split": split, "activity": activity, **summarize(group)})
    write_csv(OUT / "preprocessing_usage.csv", summary)
    gaps = [{"split": s, "activity": a, "gap_kind": k, "gap_length_frames": length, "run_count": n, "missing_joint_frames": length * n}
            for (s, a, k, length), n in sorted(gap_counts.items())]
    write_csv(OUT / "interpolation_gap_distribution.csv", gaps)
    json_dump({"status": "complete", "definitions": {
        "population": "all 1374 frozen clips before temporal sampling, including clips dropped by window protocols",
        "weighting": "frame/joint weighted percentages within split/activity; severe-clip percentage is clip weighted",
        "severe_missingness": "predeclared descriptive threshold: >25% of clip frames have >25% invalid joints",
        "invalid_joint": "unchanged preprocessing validity: confidence<0.1, nonfinite, or x=y=0",
        "invalid_shoulder_scale": "one/both raw shoulders invalid, nonfinite width, or width<=5 pixels; separate original implementation counter included",
        "torso_fallback": "unchanged _torso_scale branch, including its secondary valid-point-span/1.0 safeguards",
        "interpolated_joints": "all previously invalid joints filled by existing interpolate strategy; includes bounded interpolation and edge extension, excludes never-observed joints",
        "gaps": "run-count distribution split by bounded, leading/trailing edge and never-observed joint; no temporal resampling"},
        "by_split": {s: next(r for r in summary if r["split"] == s and r["activity"] == "ALL") for s in ("train", "val")},
        "by_split_activity": summary, "gap_distribution": gaps, "error_factor_comparisons": error_factors(results)},
        OUT / "preprocessing_usage_summary.json")
    preserve_runs()
    print("Preprocessing and error-factor diagnostics complete.", flush=True)


if __name__ == "__main__":
    main()

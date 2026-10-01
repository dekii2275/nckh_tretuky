#!/usr/bin/env python3
"""Report every predeclared experiment, including explicitly pending results."""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.window_study import EXPERIMENTS, verify_preserved, write_csv
from src.data.dataset import read_manifest
from src.metrics import classification_metrics
from src.utils import json_dump

FIELDS = "experiment sampling_mode window_length stride train_clips val_clips train_windows val_windows dropped_train_clips dropped_val_clips best_epoch early_stop_epoch best_val_window_accuracy best_val_window_macro_f1 final_clip_accuracy final_clip_macro_f1 num_batches_epoch1 epoch1_loss_mean epoch1_batch_loss_sum trainable_parameters".split()
PAIRS = [("BS -> CE", 1, 2, 5.36), ("MS -> MFS", 5, 4, 9.42), ("SAC -> DR", 6, 3, 20.39)]
ASSUMPTIONS = [
    "The conversion from variable clip length to 60 frames, stride, short-clip policy and tail policy are unreported; S1/S2/S3 are predeclared sensitivity choices.",
    "R0 uses rounded uniform source indices for long clips and linear temporal interpolation for short clips. Windows use original frame positions; full-clip missing-joint interpolation remains unchanged.",
    "Frame indices refer to the ordered OpenPose JSON files, zero based and end inclusive; no invented frames are inserted for missing JSON files.",
    "Child selection uses temporal_nearest and prefers smaller candidates on initialization; it is not manually verified child identity.",
    "Confidence threshold 0.1, unlimited missing-joint interpolation including edge filling, zero fill before normalization for never-observed joints, no smoothing, shoulder threshold 5 px and torso fallback are implementation choices.",
    "MidHip centering and shoulder normalization follow the published formulas; behavior when joints/scale are invalid is implementation-specific.",
    "Frog Pose is excluded because local directories have no usable OpenPose JSON; the paper does not explicitly explain its 10-class exclusion.",
    "The exact author session split and random seed are unreported. This study reuses the existing 1091/283 subject-session split and seed 42; subjects may occur in both sets at different sessions.",
    "Attention dropout and internal LSTM dropout are zero; Adam uses existing PyTorch defaults and zero weight decay; no scheduler, augmentation, class weighting, clipping or AMP is added.",
    "Validation accuracy is the existing early-stopping monitor with strict improvement, patience 10 and a maximum of 100 epochs. Window runs monitor window accuracy; clip aggregation is predeclared mean logits at the selected checkpoint.",
    "Window samples weight longer clips more heavily; overlapping windows are correlated. Clip evaluation excludes clips shorter than 60 frames, so coverage must accompany comparisons with R0.",
    "Epoch loss mean is sample-weighted CrossEntropyLoss; batch loss sum is the unweighted sum of batch means. R0 did not log batch sums, which cannot be reconstructed exactly from its weighted epoch mean.",
    "The paper's loss reduction and confusion-matrix normalization/evaluation unit are unreported. Reference figures here are supplied by the user, not independently verified against the paper.",
    "R0 is a legacy checkpoint without original training manifest hashes. Its current evaluation and current split are verified, but this cannot prove its historical training inputs; raw JSON contents are not hashed.",
    "Deterministic seeding retains the existing warn_only behavior and runtime defaults; bitwise reproducibility across hardware/software is not claimed.",
]


def main():
    verify_preserved()
    baseline = ROOT / "artifacts/paper_repro/verified_evaluation"
    r0 = json.loads((baseline / "validation_results.json").read_text())
    retained_ids = {r["clip_id"] for r in read_manifest(ROOT / "artifacts/splits/paper_repro/val.csv") if int(r["num_frames"]) >= 60}
    retained_predictions = [r for r in read_manifest(baseline / "predictions.csv") if r["clip_id"] in retained_ids]
    if len(retained_predictions) != len(retained_ids):
        raise ValueError("R0 retained-clip predictions incomplete")
    common_metrics, _, _ = classification_metrics([int(r["true_id"]) for r in retained_predictions],
                                                  [int(r["predicted_id"]) for r in retained_predictions], 10, r0["class_names"])
    json_dump({"experiment": "R0_clip_uniform60", "evaluation_size": len(retained_predictions),
               "population": "same 275 clips retained by every window protocol", **common_metrics},
              ROOT / "artifacts/window_audit/r0_common_clip_metrics.json")
    history = json.loads((ROOT / "runs/paper_repro/history.json").read_text())
    selected_epoch = r0["checkpoint_epoch"]
    last = history["history"][-1]["epoch"]
    rows = [{key: None for key in FIELDS}]
    rows[0].update(experiment="R0_clip_uniform60", sampling_mode="fixed_60_uniform", window_length=60,
                   train_clips=1091, val_clips=283, train_windows=1091, val_windows=283,
                   dropped_train_clips=0, dropped_val_clips=0, best_epoch=selected_epoch,
                   early_stop_epoch=last if last < 100 and last - selected_epoch >= 10 else None,
                   final_clip_accuracy=r0["accuracy"], final_clip_macro_f1=r0["macro_f1"],
                   num_batches_epoch1=math.ceil(1091 / 32), epoch1_loss_mean=history["history"][0]["train_loss"],
                   trainable_parameters=860362)
    counts = json.loads((ROOT / "artifacts/window_audit/window_summary.json").read_text())["experiments"]
    status, metrics, confusions = {}, [], []
    for experiment, level, directory in [("R0_clip_uniform60", "clip", baseline)]:
        matrix = np.loadtxt(directory / "confusion_matrix_counts.csv", delimiter=",")
        for label, i, j, reference in PAIRS:
            confusions.append({"experiment": experiment, "level": level, "pair": label,
                               "count": int(matrix[i, j]), "row_support": int(matrix[i].sum()),
                               "percentage": float(100 * matrix[i, j] / matrix[i].sum()), "paper_reference_percentage": reference})
    for stride, experiment in EXPERIMENTS.items():
        row = {key: None for key in FIELDS}
        count = counts[experiment]
        row.update(experiment=experiment, sampling_mode="sliding_window", window_length=60, stride=stride,
                   train_clips=count["train"]["clips"], val_clips=count["val"]["clips"],
                   train_windows=count["train"]["windows"], val_windows=count["val"]["windows"],
                   dropped_train_clips=count["train"]["dropped_clips"], dropped_val_clips=count["val"]["dropped_clips"],
                   trainable_parameters=860362)
        result_path = ROOT / f"artifacts/window_s{stride}/validation_results.json"
        hist_path = ROOT / f"runs/window_s{stride}/history.json"
        status[experiment] = "pending_training"
        if result_path.exists() and hist_path.exists():
            result = json.loads(result_path.read_text())
            hist = json.loads(hist_path.read_text())
            first = hist["history"][0]
            chosen = hist["selected_checkpoint_metrics"]
            if result["checkpoint_epoch"] != hist["best_epoch"]:
                raise ValueError("Evaluation/history checkpoint mismatch")
            row.update(best_epoch=hist["best_epoch"], early_stop_epoch=hist["early_stop_epoch"],
                       best_val_window_accuracy=result["window_accuracy"], best_val_window_macro_f1=result["window_macro_f1"],
                       final_clip_accuracy=result["clip_accuracy"], final_clip_macro_f1=result["clip_macro_f1"],
                       num_batches_epoch1=first["num_batches_per_epoch"], epoch1_loss_mean=first["epoch_loss_mean"],
                       epoch1_batch_loss_sum=first["epoch_batch_loss_sum"])
            if not np.isclose(chosen["val_accuracy"], result["window_accuracy"]):
                raise ValueError("Saved checkpoint evaluation differs from selected history accuracy")
            status[experiment] = "complete"
            for level in ("window", "clip"):
                metrics.append({"experiment": experiment, "level": level,
                                **{k: result[f"{level}_{k}"] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "evaluation_size")}})
                matrix = np.loadtxt(result_path.parent / f"{level}_confusion_matrix_counts.csv", delimiter=",")
                for label, i, j, reference in PAIRS:
                    support = int(matrix[i].sum())
                    confusions.append({"experiment": experiment, "level": level, "pair": label,
                                       "count": int(matrix[i, j]), "row_support": support,
                                       "percentage": float(100 * matrix[i, j] / support) if support else None,
                                       "paper_reference_percentage": reference})
        rows.append(row)
    write_csv(ROOT / "artifacts/protocol_sensitivity.csv", rows, FIELDS)
    write_csv(ROOT / "artifacts/window_audit/paper_confusion_comparison.csv", confusions)
    complete = all(s == "complete" for s in status.values())
    interpretation = "Pending: all S1/S2/S3 must finish before interpretation."
    if complete:
        gains = [r["final_clip_accuracy"] - r0["accuracy"] for r in rows[1:]]
        if max(gains) > 0.05:
            interpretation = "Sequence construction is a major sensitivity factor (>5 percentage points in clip accuracy). This does not establish the author's method or exact reproduction."
        else:
            interpretation = "These window protocols do not produce a >5-point clip-accuracy improvement. Proceed to manual child/person identity audit; do not tune the architecture."
        interpretation += " " + " ".join(
            f"{r['experiment']}: clip accuracy change {100*(r['final_clip_accuracy']-r0['accuracy']):+.2f} pp versus full R0, "
            f"{100*(r['final_clip_accuracy']-common_metrics['accuracy']):+.2f} pp versus R0 on the same retained clips; "
            f"remaining gap to 94.84% = {100*(0.9484-r['final_clip_accuracy']):.2f} pp."
            for r in rows[1:])
        if all(r["final_clip_accuracy"] < 0.80 for r in rows[1:]):
            interpretation += " All clip results remain below 80%; sequence construction does not explain the full paper gap. Proceed to manual child/person identity review, retaining the architecture and tracker until review evidence is available."
    json_dump({"status": status, "complete": complete, "metrics": metrics, "interpretation": interpretation,
               "assumptions": ASSUMPTIONS, "r0_batch_loss_sum": "unavailable_not_logged",
               "r0_common_clip_metrics": common_metrics,
               "paper_reference": {"accuracy": 0.9484, "macro_precision": 0.94, "macro_recall": 0.94, "macro_f1": 0.94}},
              ROOT / "artifacts/window_audit/study_status.json")
    lines = ["# Protocol sensitivity study", "",
             "Using the published architecture and training settings, combined with our current implementation choices for sequence construction, child selection, missing joints, and session split, reproduction R0 achieved 67.84% validation accuracy.", "",
             "All three strides were predeclared. No tested stride is asserted to be the author's true method.", "",
             "| Experiment | Train/val clips | Train/val samples | Dropped train/val clips | Best/stop epoch | Window Acc/F1 | Clip Acc/F1 |",
             "|---|---|---|---|---|---|---|"]
    def percent(value):
        return "N/A" if value is None else f"{value * 100:.4f}%"
    for row in rows:
        lines.append(f"| {row['experiment']} | {row['train_clips']}/{row['val_clips']} | {row['train_windows']}/{row['val_windows']} | {row['dropped_train_clips']}/{row['dropped_val_clips']} | {row['best_epoch']}/{row['early_stop_epoch']} | {percent(row['best_val_window_accuracy'])}/{percent(row['best_val_window_macro_f1'])} | {percent(row['final_clip_accuracy'])}/{percent(row['final_clip_macro_f1'])} |")
    lines += ["", "N/A denotes unavailable/not applicable; blank S metrics in CSV denote pending training. R0 samples are clips, not contiguous windows.", "",
              f"Coverage control: on the same {len(retained_predictions)} retained validation clips, existing R0 predictions have accuracy {percent(common_metrics['accuracy'])} and macro F1 {percent(common_metrics['macro_f1'])}. R0 was not retrained or overwritten.",
              "", "## Epoch-1 loss diagnostics", "", "| Experiment | Sample-weighted mean | Sum of batch means | Batches |", "|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['experiment']} | {row['epoch1_loss_mean']} | {row['epoch1_batch_loss_sum']} | {row['num_batches_epoch1']} |")
    lines += ["", "R0 batch count is derived from 1091 clips / batch 32. Its batch-loss sum was not logged and is not inferred from the weighted mean.", "", "## Metrics at both levels", "", "| Experiment | Level | Accuracy | Macro precision | Macro recall | Macro F1 | Support |", "|---|---|---|---|---|---|---|"]
    for m in metrics:
        lines.append(f"| {m['experiment']} | {m['level']} | {percent(m['accuracy'])} | {percent(m['macro_precision'])} | {percent(m['macro_recall'])} | {percent(m['macro_f1'])} | {m['evaluation_size']} |")
    lines += ["", "## Paper confusion diagnostics", "", "| Experiment | Level | Pair | Count/support | Percent | Supplied paper reference |", "|---|---|---|---|---|---|"]
    for c in confusions:
        lines.append(f"| {c['experiment']} | {c['level']} | {c['pair']} | {c['count']}/{c['row_support']} | {c['percentage']} | {c['paper_reference_percentage']} |")
    lines += ["", "Full per-class percentage resolution and observed off-diagonal minima: `window_audit/confusion_granularity.csv`. Larger window row supports allow finer percentage increments; this is diagnostic evidence only and does not prove that the paper used windows.", "", "## Interpretation", "", interpretation,
              "", "Paper reference supplied for comparison: validation accuracy 94.84%, macro precision/recall/F1 approximately 94%.", "", "## Remaining assumptions", ""]
    lines += [f"- {a}" for a in ASSUMPTIONS]
    review_path = ROOT / "artifacts/tracking_review/review_summary.json"
    if review_path.exists():
        review = json.loads(review_path.read_text())
        lines += ["", "## Manual identity review prepared", "",
                  f"Exported {review['review_clips']} clips across 10 classes, {review['subjects']} subjects and days D1/D8/D16. Open `tracking_review/index.html` for all-frame skeleton playback, candidate indices, selected track, bounding boxes, MidHip and confidence. This is a skeleton-only manual review, not child ground truth; RGB may be needed for ambiguous identity.",
                  "", f"Raw-candidate coverage: population single-person-only clips = {review['population_single_person_only_clips']}; population clips with some one-person frames = {review['population_clips_with_one_person_frames']}; selected clips with one-person frames = {review['clips_with_one_person_frames']}; selected clips with multi-person frames = {review['clips_with_multiple_people']}. Days: {review['per_day']}.",
                  "", "Reordering strata use a documented geometric matching proxy, not verified identity switches. The tracker and model remain unchanged. No human review labels have been assigned automatically."]
    (ROOT / "artifacts/protocol_sensitivity_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": status, "interpretation": interpretation}), flush=True)


if __name__ == "__main__":
    main()

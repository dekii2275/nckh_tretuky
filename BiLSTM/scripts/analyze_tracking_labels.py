#!/usr/bin/env python3
"""Analyze actual human A/B/C/D labels; blank labels never become negatives."""
import argparse
from collections import Counter
from pathlib import Path

from diagnostics_common import NAMES, OUT, REVIEW, ROOT, json_dump, markdown_table, preserve_runs, read_labels, read_manifest, split_records, write_csv
from src.metrics import classification_metrics


def label_rates(rows):
    counts = Counter(r["review_label"] for r in rows if r["review_label"])
    n = sum(counts.values())
    return {"clips": len(rows), "reviewed": n, "unreviewed": len(rows) - n,
            **{f"{label}_count": counts[label] for label in "ABCD"},
            **{name: counts[label] / n if n else None for label, name in zip("ABCD", ("correct_child_rate", "wrong_person_rate", "identity_switch_rate", "uncertain_rate"), strict=True)}}


def performance(rows, predictions, label):
    selected = [r for r in rows if r["review_label"] and r["review_label"] in ("BC" if label == "B+C" else label)]
    available = [predictions[r["clip_id"]] for r in selected if r["clip_id"] in predictions]
    truth, pred = [int(r["true_id"]) for r in available], [int(r["predicted_id"]) for r in available]
    metrics, _, _ = classification_metrics(truth, pred, 10, NAMES)
    # Missing groups must not be presented as zero accuracy/F1.
    return {"review_label": label, "reviewed_clips_all_splits": len(selected), "s1_validation_clips": len(available),
            "not_in_s1_validation": len(selected) - len(available), "represented_classes": len(set(truth)),
            **{k: v if available else None for k, v in metrics.items()},
            "status": "available" if available else "no_reviewed_S1_validation_clips"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=REVIEW / "manual_labels.csv")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    preserve_runs()
    rows = read_labels(args.labels)
    if len(rows) != 120:
        raise ValueError("Expected all 120 rows, with blanks for unreviewed clips")
    predictions = {r["clip_id"]: r for r in read_manifest(ROOT / "artifacts/window_s60/clip_predictions.csv")}
    overall = label_rates(rows)
    breakdown = []
    for dimension in ("activity", "session_id", "subject_id", "person_count_category"):
        def key(row):
            if dimension == "person_count_category":
                return "1 person" if int(row["max_people"]) == 1 else "2 persons" if int(row["max_people"]) == 2 else "3+ persons"
            return row[dimension]
        values = ("1 person", "2 persons", "3+ persons") if dimension == "person_count_category" else sorted({key(r) for r in rows})
        for value in values:
            breakdown.append({"dimension": dimension, "value": value, **label_rates([r for r in rows if key(r) == value])})
    write_csv(args.output_dir / "tracking_label_breakdown.csv", breakdown)
    scores = [performance(rows, predictions, label) for label in ("A", "B", "C", "D", "B+C")]
    write_csv(args.output_dir / "tracking_vs_accuracy.csv", scores)
    a, failures = scores[0], scores[-1]
    difference = a["accuracy"] - failures["accuracy"] if a["accuracy"] is not None and failures["accuracy"] is not None else None
    evidence = "manual_review_required" if not overall["reviewed"] else "insufficient_labeled_validation_groups"
    if a["s1_validation_clips"] >= 10 and failures["s1_validation_clips"] >= 10:
        evidence = "tracking_likely_major_contributor_association_only" if difference >= 0.10 else "no_large_A_vs_BC_accuracy_difference_in_reviewed_subset"
    # Conservative preparation gate: full review set completed plus user's quality thresholds.
    complete = overall["reviewed"] == len(rows)
    quality_gate = complete and overall["correct_child_rate"] >= 0.95 and overall["wrong_person_rate"] <= 0.03 and overall["identity_switch_rate"] <= 0.03
    summary = {"status": "awaiting_human_review" if not overall["reviewed"] else "partial_review" if not complete else "review_complete",
               "overall": overall, "A_minus_BC_accuracy": difference, "interpretation": evidence,
               "diagnostic_threshold": "descriptive large difference: >=10 percentage points, with at least 10 S1 validation clips in A and B+C; no causal claim",
               "person_count_category": "maximum raw detected candidates in clip (1/2/3+), not mean persons or verified identities",
               "macro_f1_definition": "macro over the same fixed 10 classes; subgroup represented-class count included",
               "population_limitation": "120 stratified/high-reordering review clips; rates do not estimate whole-dataset prevalence without a sampling design",
               "split_sensitivity_gate_passed": bool(quality_gate),
               "split_sensitivity_gate": "all 120 reviewed, correct>=95%, wrong<=3%, switches<=3%; threshold is preparation-only, not a training authorization",
               "future_seeds": [0, 1, 2, 3, 4], "candidate_tracking_strategies_prepared": False,
               "oracle_run": "not run; no confident per-frame child track annotations", "training_performed": False}
    json_dump(summary, args.output_dir / "tracking_audit_summary.json")
    lines = ["# Human tracking-label audit", "", f"Status: {summary['status']}. Reviewed {overall['reviewed']}/{len(rows)}; unreviewed {overall['unreviewed']}.", "",
             "A=correct child; B=wrong person; C=starts correctly then switches; D=uncertain. Blank is unreviewed, not D. Rates use reviewed clips including D in the denominator. No labels are inferred from model predictions, size or external binary annotations.", ""]
    lines += markdown_table([overall], ["reviewed", "unreviewed", "A_count", "B_count", "C_count", "D_count", "correct_child_rate", "wrong_person_rate", "identity_switch_rate", "uncertain_rate"])
    lines += ["", "## Saved S1 classification by human group", "", "Only held-out validation clips with saved S1 predictions are evaluated; training clips and dropped short clips are not assigned fabricated scores. Macro-F1 is over all 10 frozen classes, so class coverage matters.", ""]
    lines += markdown_table(scores, ["review_label", "reviewed_clips_all_splits", "s1_validation_clips", "represented_classes", "accuracy", "macro_f1"])
    lines += ["", f"Interpretation: {evidence}. Difference A minus B+C accuracy: {difference}.", "",
              "Breakdowns by activity/session/subject and max-person-count category are saved in tracking_label_breakdown.csv. The review set deliberately includes reordering cases; any association is descriptive and cannot establish causality.",
              "", f"Split-sensitivity preparation gate passed: {quality_gate}. No split candidates or training have been run. Do not infer tracker quality from zero human labels."]
    (args.output_dir / "tracking_audit_summary.md").write_text("\n".join(lines) + "\n")
    # Add only labels to an existing quality/error table; never change predictions.
    error_path = args.output_dir / "error_factor_analysis.csv"
    if error_path.exists():
        errors = read_manifest(error_path); mapping = {r["clip_id"]: r["review_label"] for r in rows}
        for row in errors:
            row["tracking_review_label"] = mapping.get(row["clip_id"], "")
        write_csv(error_path, errors)
        comparison_path = args.output_dir / "error_factor_comparisons.csv"
        if comparison_path.exists():
            comparisons = read_manifest(comparison_path)
            for row in comparisons:
                expected_correct = "1" if row["classification"] == "correct" else "0"
                row["human_reviewed_clips"] = sum(bool(r["tracking_review_label"]) for r in errors
                    if r["classification_correct"] == expected_correct and (row["activity"] == "ALL" or r["activity"] == row["activity"]))
            write_csv(comparison_path, comparisons)
    preserve_runs()
    print(summary["status"], overall)


if __name__ == "__main__":
    main()

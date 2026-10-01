#!/usr/bin/env python3
"""Compare supplied paper recall and saved reproduction counts, without retraining."""
import numpy as np

from diagnostics_common import NAMES, OUT, PAIRS, PAPER, ROOT, markdown_table, preserve_runs, split_records, write_csv


def matrices():
    sources = {"r0_clip": ROOT / "artifacts/paper_repro/verified_evaluation/confusion_matrix_counts.csv"}
    sources.update({f"s{s}_{level}": ROOT / f"artifacts/window_s{stride}/{level}_confusion_matrix_counts.csv"
                    for s, stride in ((1, 60), (3, 15)) for level in ("window", "clip")})
    result = {k: np.loadtxt(p, delimiter=",") for k, p in sources.items()}
    for m in result.values():
        if m.shape != (10, 10) or not np.equal(m, np.floor(m)).all() or (m < 0).any():
            raise ValueError("Invalid saved integer confusion matrix")
    return result


def main():
    preserve_runs()
    counts = matrices()
    perc = {key: 100 * m / m.sum(1, keepdims=True) for key, m in counts.items()}
    rows, contributions = [], []
    for i, name in enumerate(NAMES):
        row = {"class_name": name, "paper_recall": PAPER[name],
               **{f"{k}_recall": float(p[i, i]) for k, p in perc.items()},
               "paper_minus_s1_clip": PAPER[name] - perc["s1_clip"][i, i],
               "paper_minus_s3_clip": PAPER[name] - perc["s3_clip"][i, i],
               **{f"support_{k.replace('_clip', '') if k == 'r0_clip' else k}": int(m[i].sum()) for k, m in counts.items()}}
        rows.append(row)
        for key in ("s1_clip", "s3_clip"):
            support = counts[key][i].sum()
            gap = PAPER[name] - perc[key][i, i]
            contributions.append({"experiment_level": key, "class_name": name, "support": int(support),
                                  "recall_gap_pp": float(gap), "weighted_gap_contribution_pp": float(gap * support / counts[key].sum()),
                                  "expected_correct_count_deficit": float(gap * support / 100),
                                  "actual_errors": int(support - counts[key][i, i])})
    fields = ["class_name", "paper_recall", "r0_clip_recall", "s1_window_recall", "s1_clip_recall", "s3_window_recall", "s3_clip_recall",
              "paper_minus_s1_clip", "paper_minus_s3_clip", "support_r0", "support_s1_window", "support_s1_clip", "support_s3_window", "support_s3_clip"]
    write_csv(OUT / "per_class_gap.csv", rows, fields)
    ranked = sorted(rows, key=lambda r: max(r["paper_minus_s1_clip"], r["paper_minus_s3_clip"]), reverse=True)
    write_csv(OUT / "per_class_gap_ranked.csv", ranked, fields)
    write_csv(OUT / "per_class_gap_contributions.csv", sorted(contributions, key=lambda r: r["weighted_gap_contribution_pp"], reverse=True))
    lines = ["# Per-class recall gap", "", "All recall values are percentages; differences are percentage points. Paper reference diagonals are supplied by the user. No cause is inferred.", "",
             "Ranking uses the larger of the S1/S3 clip recall gaps. S1/S3 have 275 validation clips, R0 has 283; supports are included in the CSV. Small-support classes need cautious interpretation.", ""]
    lines += markdown_table(ranked, ["class_name", "paper_recall", "s1_clip_recall", "s3_clip_recall", "paper_minus_s1_clip", "paper_minus_s3_clip", "support_s1_clip"])
    for key in ("s1_clip", "s3_clip"):
        top = sorted([r for r in contributions if r["experiment_level"] == key], key=lambda r: r["weighted_gap_contribution_pp"], reverse=True)
        near = [r["class_name"] for r in rows if abs(r[f"paper_minus_{key}"]) <= 5]
        weak = [r["class_name"] for r in rows if r[f"paper_minus_{key}"] >= 20]
        lines += ["", f"## {key}", "", "Largest weighted contributions: " + ", ".join(r["class_name"] for r in top[:5]) + ".",
                  "Within 5 percentage points of paper recall: " + (", ".join(near) or "none") + ".",
                  "Recall shortfall at least 20 points: " + (", ".join(weak) or "none") + ".", ""]
        lines += markdown_table(top, ["class_name", "support", "recall_gap_pp", "weighted_gap_contribution_pp", "actual_errors"])
    lines += ["", "Weighted contributions apply the paper recalls to our validation class distribution. Their sum is the gap to that hypothetical weighted recall, not a decomposition of the published 94.84% accuracy: the paper's class supports and evaluation population are not given here."]
    (OUT / "per_class_gap.md").write_text("\n".join(lines) + "\n")

    comparisons = []
    for (i, j), reference in PAIRS.items():
        comparisons.append({"class_true": NAMES[i], "class_pred": NAMES[j], "paper_percent": reference,
                            **{("r0_percent" if k == "r0_clip" else f"{k}_percent"): float(p[i, j]) for k, p in perc.items()}})
    write_csv(OUT / "confusion_comparison.csv", comparisons)
    top_rows = []
    for key in ("s1_window", "s1_clip", "s3_window", "s3_clip"):
        pairs = [{"experiment_level": key, "class_true": NAMES[i], "class_pred": NAMES[j],
                  "count": int(counts[key][i, j]), "row_support": int(counts[key][i].sum()),
                  "percent": float(perc[key][i, j]), "paper_percent": PAIRS.get((i, j)),
                  "reference_available": (i, j) in PAIRS}
                 for i in range(10) for j in range(10) if i != j and counts[key][i, j] > 0]
        pairs.sort(key=lambda r: (-r["count"], -r["percent"], r["class_true"], r["class_pred"]))
        top_rows.extend({"rank_by_error_count": rank, **r} for rank, r in enumerate(pairs[:20], 1))
    write_csv(OUT / "top20_confusion_pairs.csv", top_rows)
    lines = ["# Confusion structure", "", "Paper off-diagonal reference is available for only three pairs. Similarity on those pairs cannot establish similarity of the full confusion matrix or exact reproduction.", ""]
    lines += markdown_table(comparisons, list(comparisons[0]))
    for key in ("s1_window", "s1_clip", "s3_window", "s3_clip"):
        lines += ["", f"## {key}: top 20 nonzero pairs by error count", ""]
        lines += markdown_table([r for r in top_rows if r["experiment_level"] == key], ["rank_by_error_count", "class_true", "class_pred", "count", "row_support", "percent", "paper_percent"])
    lines += ["", "S3's three supplied pairs are close in absolute percentage, but other large reproduction errors must be examined separately. Missing paper off-diagonal entries are marked N/A, not treated as zero."]
    (OUT / "confusion_structure.md").write_text("\n".join(lines) + "\n")

    splits = split_records()
    distribution = []
    for name in NAMES:
        groups = {s: [r for r in records if r["activity"] == name] for s, records in splits.items()}
        total = sum(len(v) for v in groups.values())
        row = {"class": name, "total": total, "train_count": len(groups["train"]), "val_count": len(groups["val"]),
               "train_percent": 100 * len(groups["train"]) / total, "val_percent": 100 * len(groups["val"]) / total}
        for s, group in groups.items():
            row.update({f"{s}_subjects": len({r["subject_id"] for r in group}), f"{s}_session_days": len({r["session_id"] for r in group}),
                        f"{s}_subject_session_groups": len({(r["subject_id"], r["session_id"]) for r in group}),
                        f"{s}_subject_ids": ";".join(sorted({r["subject_id"] for r in group})),
                        f"{s}_session_ids": ";".join(sorted({r["session_id"] for r in group})),
                        f"{s}_class_share_percent": 100 * len(group) / len(splits[s]),
                        f"{s}_retained_clips": sum(int(r["num_frames"]) >= 60 for r in group),
                        f"{s}_s1_windows": sum(max(0, (int(r["num_frames"]) - 60) // 60 + 1) for r in group)})
        distribution.append(row)
    write_csv(OUT / "current_split_class_distribution.csv", distribution)
    lines = ["# Current frozen split composition", "", "Percentages train_percent/val_percent are within each class; class_share_percent is within each split. Counts include short clips; retained clip and S1-window counts are separate.", ""]
    lines += markdown_table(distribution, ["class", "total", "train_count", "val_count", "val_percent", "train_subjects", "val_subjects", "train_subject_session_groups", "val_subject_session_groups"])
    lines += ["", "Uneven class allocation, few held-out subjects and very small validation supports can make class estimates unstable. This is a descriptive composition finding, not proof that the split causes the observed errors.",
              "", "Future seed set is exactly [0, 1, 2, 3, 4], S1 stride 60. No candidate splits or training configs are generated until human tracking review meets the quality gate. No seed was tested or selected."]
    (OUT / "current_split_class_distribution.md").write_text("\n".join(lines) + "\n")
    preserve_runs()
    print("Per-class, confusion and current split diagnostics saved.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnose row percentage resolution; this cannot identify the author protocol."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.window_study import EXPERIMENTS, write_csv
from src.utils import json_dump


def granularity(matrix, names):
    matrix = np.asarray(matrix)
    if matrix.shape != (len(names), len(names)) or np.any(matrix < 0) or not np.equal(matrix, np.floor(matrix)).all():
        raise ValueError("Expected a nonnegative integer confusion matrix")
    rows = []
    for i, name in enumerate(names):
        count = int(matrix[i].sum())
        off = np.delete(matrix[i], i)
        observed = off[off > 0]
        rows.append({"activity": name, "row_sample_count": count,
                     "smallest_possible_nonzero_percentage_step": 100.0 / count if count else None,
                     "smallest_observed_nonzero_off_diagonal_percentage": float(observed.min() * 100.0 / count) if len(observed) else None})
    return rows


def main():
    baseline = ROOT / "artifacts/paper_repro/verified_evaluation"
    names = json.loads((baseline / "validation_results.json").read_text())["class_names"]
    sources = [("R0_clip_uniform60", "clip", baseline / "confusion_matrix_counts.csv")]
    sources += [(experiment, level, ROOT / f"artifacts/window_s{stride}/{level}_confusion_matrix_counts.csv")
                for stride, experiment in EXPERIMENTS.items() for level in ("window", "clip")]
    rows, status = [], []
    for experiment, level, path in sources:
        if not path.exists():
            status.append({"experiment": experiment, "level": level, "status": "pending_training"})
            continue
        items = granularity(np.loadtxt(path, delimiter=","), names)
        rows.extend({"experiment": experiment, "level": level, **r} for r in items)
        observed = [r["smallest_observed_nonzero_off_diagonal_percentage"] for r in items
                    if r["smallest_observed_nonzero_off_diagonal_percentage"] is not None]
        status.append({"experiment": experiment, "level": level, "status": "complete",
                       "smallest_observed_nonzero_off_diagonal_percentage": min(observed) if observed else None})
    output = ROOT / "artifacts/window_audit"
    write_csv(output / "confusion_granularity.csv", rows)
    json_dump({"matrices": status, "rows": rows,
               "interpretation": "Diagnostic percentage resolution only; not proof that the paper used windows."},
              output / "confusion_granularity.json")


if __name__ == "__main__":
    main()

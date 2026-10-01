#!/usr/bin/env python3
"""Independent checks of saved predictions, aggregation, matrices and run selection."""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import read_manifest, window_metadata
from src.metrics import classification_metrics
from src.reproducibility import sha256_file
from src.utils import json_dump, load_config
from src.window_study import EXPERIMENTS, frozen_inputs, verify_preserved


def main():
    protected = verify_preserved()
    checked = []
    for stride, experiment in EXPERIMENTS.items():
        config = load_config(ROOT / f"configs/window_s{stride}.yaml")
        splits = frozen_inputs(config)
        output = ROOT / f"artifacts/window_s{stride}"
        result = json.loads((output / "validation_results.json").read_text())
        history = json.loads((ROOT / f"runs/window_s{stride}/history.json").read_text())
        assert result["checkpoint_sha256"] == sha256_file(ROOT / f"checkpoints/window_s{stride}/best.pt")
        assert result["checkpoint_verification"]["status"] == "recorded_manifests_verified"
        assert result["checkpoint_verification"]["source_matches"]
        assert history["environment"]["number_of_trainable_parameters"] == 860362
        selected = max(history["history"], key=lambda r: r["val_accuracy"])
        assert selected["epoch"] == history["best_epoch"] == result["checkpoint_epoch"]
        if history["early_stop_epoch"] is not None:
            assert history["early_stop_epoch"] == history["best_epoch"] + 10
        expected_train = window_metadata(splits["train"], config["sequence"])
        for row in history["history"]:
            assert row["num_batches_per_epoch"] == math.ceil(len(expected_train) / 32)
            assert row["epoch_loss_mean"] == row["train_loss"]
            assert all(np.isfinite(v) for v in row.values())
        windows = read_manifest(output / "window_predictions.csv")
        clips = read_manifest(output / "clip_predictions.csv")
        expected_val = window_metadata(splits["val"], config["sequence"])
        assert len(windows) == len(expected_val)
        assert len({r["window_id"] for r in windows}) == len(windows)
        for actual, expected in zip(windows, expected_val, strict=True):
            assert all(str(actual[k]) == str(v) for k, v in expected.items() if k != "clip_index")
        logits = np.asarray([[float(r[f"logit_{i}"]) for i in range(10)] for r in windows])
        assert np.isfinite(logits).all()
        assert np.array_equal(logits.argmax(1), [int(r["predicted_id"]) for r in windows])
        ids = np.asarray([r["clip_id"] for r in windows])
        assert {r["clip_id"] for r in clips} == set(ids)
        assert len(clips) == len(set(ids)) == 275
        for clip in clips:
            means = logits[ids == clip["clip_id"]].mean(0)
            assert int(clip["num_windows"]) == int((ids == clip["clip_id"]).sum())
            assert np.allclose(means, [float(clip[f"mean_logit_{i}"]) for i in range(10)], atol=1e-12)
            assert int(means.argmax()) == int(clip["predicted_id"])
        for level, rows in (("window", windows), ("clip", clips)):
            metrics, _, counts = classification_metrics([int(r["true_id"]) for r in rows],
                                                         [int(r["predicted_id"]) for r in rows], 10, result["class_names"])
            for key, value in metrics.items():
                assert np.isclose(value, result[f"{level}_{key}"], atol=1e-12)
            assert np.array_equal(counts, np.loadtxt(output / f"{level}_confusion_matrix_counts.csv", delimiter=","))
            support = counts.sum(1, keepdims=True)
            percent = np.divide(100.0 * counts, support, out=np.zeros_like(counts, dtype=float), where=support != 0)
            assert np.allclose(percent, np.loadtxt(output / f"{level}_confusion_matrix_percent.csv", delimiter=","), atol=1e-9)
        checked.append({"experiment": experiment, "status": "pass", "train_windows": len(expected_train),
                        "val_windows": len(windows), "val_clips": len(clips), "best_epoch": selected["epoch"]})
    json_dump({"status": "pass", "experiments": checked, "protected_files_unchanged": len(protected),
               "checks": ["frozen config and split", "no leakage before windows", "complete metadata and coverage",
                          "saved logits and argmax", "mean-logit aggregation", "both metric levels and matrices",
                          "loss diagnostics", "strict accuracy-based selection and patience", "checkpoint/source provenance"]},
              ROOT / "artifacts/window_audit/final_verification.json")
    print(json.dumps(checked, indent=2))


if __name__ == "__main__":
    main()

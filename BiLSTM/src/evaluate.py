"""Evaluate one selected checkpoint on the held-out test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from .data.dataset import read_manifest
from .engine import build_model, class_names_from_records, make_loader, run_epoch
from .utils import cfg_path, ensure_dir, json_dump, load_config
from .reproducibility import capture_provenance, sha256_file, verify_checkpoint_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Write a separate evaluation without replacing existing results")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Evaluation batch size; defaults to evaluation.batch_size or 1 for full-clip attention.",
    )
    parser.add_argument(
        "--split",
        choices=("test", "val"),
        default=None,
        help="Split to evaluate; defaults to evaluation.split or test when available.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    is_window = (
        config.get("sequence", {}).get("mode") == "sliding_window"
        and str(config.get("experiment", "")).startswith("WINDOW_")
    )
    from .sequence_study import is_task3_config, save_task3_evaluation, validate_task3_config
    is_task3 = is_task3_config(config)
    if is_task3:
        validate_task3_config(config)
    if is_window:
        from .window_study import frozen_inputs, parameter_breakdown, save_window_evaluation
        frozen_inputs(config)
    artifacts = cfg_path(config, "outputs", "artifacts_dir", default="artifacts")
    output_dir = args.output_dir.resolve() if args.output_dir else artifacts
    split_dir = cfg_path(config, "outputs", "split_dir", default="artifacts/splits")
    evaluation_config = config.get("evaluation", {})
    requested_split = args.split or evaluation_config.get("split")
    if requested_split is None:
        requested_split = "test" if (split_dir / "test.csv").exists() else "val"
    records = read_manifest(split_dir / f"{requested_split}.csv")
    if not records:
        raise RuntimeError(f"{requested_split} manifest is empty or missing")
    manifest_file = artifacts / "manifest.csv"
    if not manifest_file.is_file():
        manifest_file = Path(config["_project_root"]) / "artifacts/paper_repro/manifest.csv"
    manifest_records = read_manifest(manifest_file)
    class_names = class_names_from_records(manifest_records)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    provenance = capture_provenance(config)
    verification = verify_checkpoint_inputs(checkpoint, config, provenance)
    print(json.dumps({"checkpoint_verification": verification}, ensure_ascii=False))
    saved_names = checkpoint.get("class_names")
    if saved_names:
        class_names = list(saved_names)
    model = build_model(config, len(class_names))
    if is_window:
        parameter_breakdown(model)
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    evaluation_batch_size = args.batch_size or int(evaluation_config.get("batch_size", 1))
    if evaluation_batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    from .data.sequence import SequenceBuilder
    builder = SequenceBuilder(dict(config.get("sequence", {})))
    is_multi_window = builder.is_multi_window
    loader = make_loader(records, config, train=False, batch_size=evaluation_batch_size)
    with torch.no_grad():
        stats = run_epoch(model, loader, criterion, device, class_names,
                          collect_predictions=is_window or is_task3 or is_multi_window)
    if is_window:
        result = save_window_evaluation(output_dir, stats, class_names)
        result.update({"experiment": config["experiment"], "class_names": class_names,
                       "checkpoint_epoch": checkpoint["epoch"], "checkpoint_sha256": sha256_file(args.checkpoint),
                       "checkpoint_verification": verification, "provenance": provenance,
                       "frozen_val_clips": len(records), "dropped_val_clips": len(records) - result["clip_evaluation_size"]})
        json_dump(result, output_dir / "validation_results.json")
        print(json.dumps({k: v for k, v in result.items() if k != "provenance"}, indent=2))
        return
    if is_task3 or is_multi_window:
        from .sequence_study import save_task3_evaluation
        result = save_task3_evaluation(output_dir, stats, class_names)
        result.update({
            "experiment": config.get("experiment", "multi_window"),
            "class_names": class_names,
            "accuracy": result["clip_accuracy"],
            "macro_f1": result["clip_macro_f1"],
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "checkpoint_verification": verification,
            "provenance": provenance,
            "split": requested_split,
            "evaluation_unit_primary": "source clip",
        })
        json_dump(result, output_dir / "validation_results.json")
        print(json.dumps({key: value for key, value in result.items() if key != "provenance"}, indent=2))
        return
    from .metrics import classification_metrics

    overall, per_class, matrix = classification_metrics(
        stats["y_true"], stats["y_pred"], len(class_names), class_names
    )
    result = {
        **overall,
        "loss": stats["loss"],
        "checkpoint": str(args.checkpoint.resolve()),
        "split": requested_split,
        "evaluation_size": len(records),
        "test_size": len(records) if requested_split == "test" else None,
        "evaluation_unit": "clip",
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_verification": verification,
        "provenance": provenance,
        "class_names": class_names,
        "reference_target_note": "Paper-reported validation Accuracy ~94.84% and Macro-F1 ~0.94 are reference targets, not guaranteed results.",
    }
    result_filename = "validation_results.json" if requested_split == "val" else "results.json"
    ensure_dir(output_dir)
    json_dump(result, output_dir / result_filename)
    with (output_dir / "per_class_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_class[0].keys()))
        writer.writeheader()
        writer.writerows(per_class)
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clip_id", "subject_id", "session_id", "true_id", "predicted_id"])
        writer.writeheader()
        for record, truth, prediction in zip(records, stats["y_true"], stats["y_pred"], strict=True):
            writer.writerow({
                "clip_id": record["clip_id"], "subject_id": record["subject_id"],
                "session_id": record["session_id"], "true_id": truth, "predicted_id": prediction,
            })
    np.savetxt(output_dir / "confusion_matrix_counts.csv", matrix, delimiter=",", fmt="%d")
    support = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(100.0 * matrix, support, out=np.zeros_like(matrix, dtype=float), where=support != 0)
    np.savetxt(output_dir / "confusion_matrix_percent.csv", percentages, delimiter=",", fmt="%.6f")
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=f"MMASD {requested_split} confusion matrix (clip counts)",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2.0 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column, row, int(matrix[row, column]),
                ha="center", va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved {output_dir / result_filename}")


if __name__ == "__main__":
    main()

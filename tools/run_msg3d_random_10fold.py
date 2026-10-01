#!/usr/bin/env python3
"""Round 5: deterministic 10-fold random-reproduction-like plain MSG3D.

The experiment contract is intentionally frozen to the successful Round 4
configuration.  The runner validates all folds before training any fold and
does not evaluate the official test split during training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from run_round4_fold00 import (  # noqa: E402
    ARTIFACT_ROOT,
    BASE_LR,
    BATCH_SIZE,
    CHECKPOINT,
    EPOCHS,
    FORWARD_BATCH_SIZE,
    GAMMA,
    MILESTONES,
    MOMENTUM,
    NESTEROV,
    TEST_BATCH_SIZE,
    WEIGHT_DECAY,
    build_loaded_model,
    evaluate,
    evaluate_subset_accuracy,
    make_loader,
    parameter_report,
    run_batch_backward,
    run_gradient_diagnostic,
    run_optimizer_diagnostic,
    save_confusion_csv,
    sentinel_exposure,
    set_seed,
    snapshot_code_state,
    train_epoch,
    write_json,
)


PROTOCOL = "random_reproduction_like"
BASE_SEED = 2026091704
FOLD_COUNT = 10
EXPECTED_DUPLICATE_OVERLAPS = [24, 12, 6, 12, 13, 0, 6, 7, 12, 13]
DEFAULT_OUTPUT_ROOT = ARTIFACT_ROOT / "experiments" / "round5_random_10fold"
SENTINEL_SUBJECTS = ("ASD_004", "ASD_008", "ASD_022")


def json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest() -> list[dict[str, str]]:
    with (ARTIFACT_ROOT / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_fold(fold_id: int) -> dict[str, object]:
    path = ARTIFACT_ROOT / "folds" / PROTOCOL / f"fold_{fold_id:02d}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def fold_integrity(rows: list[dict[str, str]], fold: dict[str, object], fold_id: int) -> dict[str, object]:
    if fold.get("protocol") != PROTOCOL or int(fold["fold_id"]) != fold_id:
        raise AssertionError(f"fold_{fold_id:02d}: wrong protocol or fold id")
    if int(fold["train_sample_count"]) != 720 or int(fold["test_sample_count"]) != 80:
        raise AssertionError(f"fold_{fold_id:02d}: expected 720/80 samples")
    if int(fold["train_subject_count"]) != 90 or int(fold["test_subject_count"]) != 10:
        raise AssertionError(f"fold_{fold_id:02d}: expected 90/10 subjects")

    by_sample = {row["sample_id"]: row for row in rows}
    train_ids = set(fold["train_sample_ids"])
    test_ids = set(fold["test_sample_ids"])
    train_subjects = set(fold["train_subject_ids"])
    test_subjects = set(fold["test_subject_ids"])
    if train_ids & test_ids:
        raise AssertionError(f"fold_{fold_id:02d}: sample leakage")
    if train_subjects & test_subjects:
        raise AssertionError(f"fold_{fold_id:02d}: subject leakage")
    if train_ids | test_ids != set(by_sample):
        raise AssertionError(f"fold_{fold_id:02d}: sample IDs do not cover manifest")

    for subject_id in train_subjects | test_subjects:
        partitions = {
            "train" if row["sample_id"] in train_ids else "test"
            for row in rows
            if row["subject_id"] == subject_id
        }
        if partitions != {"train"} and partitions != {"test"}:
            raise AssertionError(f"fold_{fold_id:02d}: augmentation grouping leakage: {subject_id}")

    class_counts = {
        "train_ASD": sum(by_sample[sample]["class_name"] == "ASD" for sample in train_ids),
        "train_TD": sum(by_sample[sample]["class_name"] == "TD" for sample in train_ids),
        "test_ASD": sum(by_sample[sample]["class_name"] == "ASD" for sample in test_ids),
        "test_TD": sum(by_sample[sample]["class_name"] == "TD" for sample in test_ids),
    }
    if class_counts != {"train_ASD": 360, "train_TD": 360, "test_ASD": 40, "test_TD": 40}:
        raise AssertionError(f"fold_{fold_id:02d}: class counts mismatch: {class_counts}")

    train_hashes = {by_sample[sample]["numeric_hash"] for sample in train_ids if by_sample[sample]["numeric_hash"]}
    test_hashes = {by_sample[sample]["numeric_hash"] for sample in test_ids if by_sample[sample]["numeric_hash"]}
    duplicate_overlap = len(train_hashes & test_hashes)
    if duplicate_overlap != EXPECTED_DUPLICATE_OVERLAPS[fold_id]:
        raise AssertionError(
            f"fold_{fold_id:02d}: duplicate overlap {duplicate_overlap}, "
            f"expected {EXPECTED_DUPLICATE_OVERLAPS[fold_id]}"
        )
    if int(fold["duplicate_leakage_count"]) != duplicate_overlap:
        raise AssertionError(f"fold_{fold_id:02d}: fold metadata duplicate count mismatch")

    return {
        "fold": fold_id,
        "train_samples": len(train_ids),
        "test_samples": len(test_ids),
        "train_subjects": len(train_subjects),
        "test_subjects": len(test_subjects),
        **class_counts,
        "duplicate_hash_overlap": duplicate_overlap,
        "subject_leakage": False,
        "sample_leakage": False,
        "augmentation_leakage": False,
    }


def validate_all_folds(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summaries = []
    sentinel_map = []
    for fold_id in range(FOLD_COUNT):
        fold = read_fold(fold_id)
        summaries.append(fold_integrity(rows, fold, fold_id))
        exposure = sentinel_exposure(rows, fold)
        placement = {
            subject: "TEST" if subject in exposure["test_subjects"] else "TRAIN"
            for subject in SENTINEL_SUBJECTS
        }
        sentinel_map.append(
            {
                "fold": fold_id,
                **placement,
                "test_sentinel_samples": sum(
                    int(row["sentinel_count"])
                    for row in rows
                    if row["sample_id"] in set(fold["test_sample_ids"])
                    and int(row["sentinel_count"]) > 0
                ),
                "test_sentinel_subjects": exposure["test_subjects"],
            }
        )
    actual = [int(row["duplicate_hash_overlap"]) for row in summaries]
    if actual != EXPECTED_DUPLICATE_OVERLAPS:
        raise AssertionError(f"all-fold duplicate overlap mismatch: {actual}")
    return summaries, sentinel_map


def frozen_config(seed_state: dict[str, object], code_state: dict[str, object], seeds: list[int]) -> dict[str, object]:
    return {
        "round": 5,
        "protocol": PROTOCOL,
        "folds": list(range(FOLD_COUNT)),
        "base_seed": BASE_SEED,
        "seeds": seeds,
        "data_contract": "DATA_CONTRACT_v1",
        "preprocessing": "DRYAD_MSG3D_PREPROCESS_V1",
        "class_mapping": {"TD": 0, "ASD": 1},
        "input_shape": [3, 60, 25, 2],
        "coordinate_order": "XYZ_HIGH_CONFIDENCE",
        "temporal_policy": "right_zero_pad_T60",
        "person_policy": "person0_actual_person1_zero",
        "sentinel_policy": "preserve_value_100_no_repair",
        "architecture": "plain MSG3D",
        "num_class": 2,
        "num_point": 25,
        "num_person": 2,
        "dream": False,
        "use_angle": False,
        "skepxel": False,
        "vit": False,
        "auxiliary_loss": False,
        "checkpoint_source": str(CHECKPOINT),
        "checkpoint_policy": "final_epoch_30_only",
        "epochs": EPOCHS,
        "optimizer": "SGD",
        "learning_rate": BASE_LR,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "nesterov": NESTEROV,
        "scheduler": "MultiStepLR",
        "milestones": MILESTONES,
        "gamma": GAMMA,
        "scheduler_step_timing": "end_of_epoch",
        "batch_size": BATCH_SIZE,
        "forward_batch_size": FORWARD_BATCH_SIZE,
        "test_batch_size": TEST_BATCH_SIZE,
        "drop_last": True,
        "loss": "CrossEntropyLoss",
        "test_evaluation_during_training": False,
        "validation": False,
        "seed_state_example": seed_state,
        "code_state": code_state,
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


def write_immutable_config(path: Path, config: dict[str, object]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError(f"frozen config differs from existing file: {path}")
        return
    write_json(path, config)


def subset_metrics(results: list[dict[str, object]]) -> dict[str, object]:
    if not results:
        return {"count": 0, "accuracy": None, "confusion_matrix": None, "sample_ids": []}
    labels = np.asarray([int(row["true_label"]) for row in results], dtype=np.int64)
    predictions = np.asarray([int(row["predicted_label"]) for row in results], dtype=np.int64)
    confusion = np.zeros((2, 2), dtype=np.int64)
    for true, predicted in zip(labels, predictions):
        confusion[true, predicted] += 1
    return {
        "count": len(results),
        "accuracy": float((labels == predictions).mean()),
        "confusion_matrix": confusion.tolist(),
        "sample_ids": [row["sample_id"] for row in results],
    }


def run_fold(
    fold_id: int,
    seed: int,
    rows: list[dict[str, str]],
    output_root: Path,
    device: torch.device,
    code_state: dict[str, object],
    split_summary: dict[str, object],
    resume: bool,
    overwrite: bool,
) -> dict[str, object]:
    fold = read_fold(fold_id)
    fold_root = output_root / f"fold_{fold_id:02d}"
    metadata_path = fold_root / "run_metadata.json"
    if metadata_path.exists() and resume and not overwrite:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("training_health") == "HEALTHY" and existing.get("seed") == seed:
            print(f"FOLD_{fold_id:02d}_SKIP existing healthy result")
            return existing
        raise FileExistsError(f"incomplete/incompatible fold exists: {fold_root}; use --overwrite")
    if fold_root.exists() and any(fold_root.iterdir()) and not overwrite:
        raise FileExistsError(f"fold exists: {fold_root}; use --overwrite")
    fold_root.mkdir(parents=True, exist_ok=True)

    seed_state = set_seed(seed)
    loader_root = ARTIFACT_ROOT / "loader" / PROTOCOL
    train_data = loader_root / f"train_data_fold_{fold_id:02d}.npy"
    train_labels = loader_root / f"train_labels_fold_{fold_id:02d}.pkl"
    test_data = loader_root / f"test_data_fold_{fold_id:02d}.npy"
    test_labels = loader_root / f"test_labels_fold_{fold_id:02d}.pkl"
    noaug_root = ARTIFACT_ROOT / "noaug" / PROTOCOL
    noaug_data = noaug_root / f"test_data_NoAug_fold_{fold_id:02d}.npy"
    noaug_labels = noaug_root / f"test_labels_NoAug_fold_{fold_id:02d}.pkl"

    diagnostic_loader = make_loader(train_data, train_labels, BATCH_SIZE, shuffle=False, seed=seed + 1)
    diagnostic_batch = next(iter(diagnostic_loader))
    parameter_model, load_info = build_loaded_model(device)
    parameters = parameter_report(parameter_model)
    write_json(fold_root / "parameter_report.json", {**parameters, "checkpoint_load": load_info})
    del parameter_model
    gradient = run_gradient_diagnostic(device, diagnostic_batch)
    optimizer_diag = run_optimizer_diagnostic(device, diagnostic_batch)
    write_json(fold_root / "gradient_diagnostic.json", gradient)
    write_json(fold_root / "optimizer_step_diagnostic.json", optimizer_diag)

    model, load_info = build_loaded_model(device)
    optimizer = SGD(
        model.parameters(), lr=BASE_LR, momentum=MOMENTUM,
        nesterov=NESTEROV, weight_decay=WEIGHT_DECAY,
    )
    scheduler = MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA)
    criterion = nn.CrossEntropyLoss()
    checkpoint_dir = fold_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "epoch_00.pt")
    log_path = fold_root / "training_log.csv"
    fields = [
        "fold", "epoch", "seed", "learning_rate", "learning_rate_after_scheduler",
        "train_loss", "train_accuracy", "samples_seen", "predicted_TD_count",
        "predicted_ASD_count", "elapsed_seconds",
    ]
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for epoch in range(1, EPOCHS + 1):
            start = time.perf_counter()
            epoch_loader = make_loader(train_data, train_labels, BATCH_SIZE, shuffle=True, seed=seed + epoch)
            learning_rate = float(optimizer.param_groups[0]["lr"])
            train_metrics = train_epoch(model, optimizer, criterion, epoch_loader, device)
            scheduler.step()
            row = {
                "fold": fold_id,
                "epoch": epoch,
                "seed": seed,
                "learning_rate": learning_rate,
                "learning_rate_after_scheduler": float(optimizer.param_groups[0]["lr"]),
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "samples_seen": train_metrics["samples"],
                "predicted_TD_count": train_metrics["predicted_td"],
                "predicted_ASD_count": train_metrics["predicted_asd"],
                "elapsed_seconds": time.perf_counter() - start,
            }
            writer.writerow(row)
            handle.flush()
            print("TRAIN", json.dumps(row, sort_keys=True))
            if epoch in {10, 20, 30}:
                torch.save(model.state_dict(), checkpoint_dir / f"epoch_{epoch:02d}.pt")

    final_model = fold_root / "final_model.pt"
    torch.save(model.state_dict(), final_model)
    torch.save(optimizer.state_dict(), fold_root / "optimizer_state.pt")
    torch.save(scheduler.state_dict(), fold_root / "scheduler_state.pt")

    by_path = {row["source_path"]: row for row in rows}
    test_loader = make_loader(test_data, test_labels, TEST_BATCH_SIZE, shuffle=False, seed=seed + 100)
    full_metrics, full_results, full_logits, full_predictions = evaluate(model, test_loader, device, by_path)
    write_json(fold_root / "test_predictions.json", full_results)
    save_confusion_csv(fold_root / "confusion_matrix.csv", full_metrics["confusion_matrix"])

    noaug_loader = make_loader(noaug_data, noaug_labels, TEST_BATCH_SIZE, shuffle=False, seed=seed + 101)
    noaug_metrics, noaug_results, _noaug_logits, _noaug_predictions = evaluate(model, noaug_loader, device, by_path)
    write_json(fold_root / "noaug_metrics.json", noaug_metrics)

    train_hashes = {
        rows[index]["numeric_hash"] for index in fold["train_indices"] if rows[index]["numeric_hash"]
    }
    duplicated = [item for item in full_results if item["numeric_hash"] in train_hashes]
    nonduplicated = [item for item in full_results if item["numeric_hash"] not in train_hashes]
    duplicate_analysis = {
        "train_test_numeric_hash_overlap_count": int(fold["duplicate_leakage_count"]),
        "duplicated_in_train_test_subset": subset_metrics(duplicated),
        "nonduplicated_test_subset": subset_metrics(nonduplicated),
    }
    write_json(fold_root / "duplicate_analysis.json", duplicate_analysis)

    sentinel_results = [item for item in full_results if int(item["sentinel_count"]) > 0]
    sentinel_original = [item for item in sentinel_results if by_path[item["source_path"]]["is_original"] == "1"]
    sentinel_augmented = [item for item in sentinel_results if by_path[item["source_path"]]["is_original"] != "1"]
    sentinel_analysis = {
        "test_samples": sentinel_results,
        "original_samples": sentinel_original,
        "augmentation_samples": sentinel_augmented,
        "original_subset": subset_metrics(sentinel_original),
        "augmentation_subset": subset_metrics(sentinel_augmented),
    }
    write_json(fold_root / "sentinel_analysis.json", sentinel_analysis)

    reload_model, reload_load_info = build_loaded_model(device)
    reload_model.load_state_dict(torch.load(final_model, map_location=device))
    reload_metrics, _reload_results, reload_logits, reload_predictions = evaluate(
        reload_model, test_loader, device, by_path
    )
    max_logit_difference = float(np.max(np.abs(full_logits - reload_logits)))
    prediction_equality = bool(np.array_equal(full_predictions, reload_predictions))
    if not prediction_equality or max_logit_difference > 0.0:
        raise AssertionError(
            f"fold_{fold_id:02d}: checkpoint reload mismatch, "
            f"equal={prediction_equality}, max_diff={max_logit_difference}"
        )

    with log_path.open(newline="", encoding="utf-8") as handle:
        training_rows = list(csv.DictReader(handle))
    losses = [float(row["train_loss"]) for row in training_rows]
    accuracies = [float(row["train_accuracy"]) for row in training_rows]
    health = (
        len(training_rows) == EPOCHS
        and all(math.isfinite(value) for value in losses + accuracies)
        and abs(losses[-1] - losses[0]) > 1e-8
        and all(
            int(row["predicted_TD_count"]) > 0 and int(row["predicted_ASD_count"]) > 0
            for row in training_rows[-3:]
        )
        and prediction_equality
        and math.isfinite(max_logit_difference)
        and gradient["all_gradients_finite"]
    )

    round4_reference = None
    if fold_id == 0:
        reference_metrics = ARTIFACT_ROOT / "experiments" / "round4_fold00" / "test_metrics.json"
        if reference_metrics.exists():
            reference = json.loads(reference_metrics.read_text(encoding="utf-8"))
            round4_reference = {
                "accuracy": reference["full"]["accuracy"],
                "macro_f1": reference["full"]["macro_f1"],
                "accuracy_difference": full_metrics["accuracy"] - reference["full"]["accuracy"],
                "macro_f1_difference": full_metrics["macro_f1"] - reference["full"]["macro_f1"],
            }

    metadata = {
        "round": 5,
        "protocol": PROTOCOL,
        "fold": fold_id,
        "seed": seed,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(device) if torch.cuda.is_available() else None,
        "code_state": code_state,
        "split": split_summary,
        "seed_state": seed_state,
        "model": {
            "architecture": "plain MSG3D",
            "num_class": 2,
            "num_point": 25,
            "num_person": 2,
            "dream": False,
            "use_angle": False,
            "checkpoint_load": load_info,
            "parameter_report": parameters,
        },
        "training_health": "HEALTHY" if health else "SUSPICIOUS",
        "training_performed": True,
        "test_evaluation_during_training": False,
        "full_test": full_metrics,
        "noaug_test": noaug_metrics,
        "duplicate_analysis": duplicate_analysis,
        "sentinel_analysis": sentinel_analysis,
        "checkpoint_reload": {
            "prediction_equality": prediction_equality,
            "max_logit_difference": max_logit_difference,
            "checkpoint_load": reload_load_info,
            "reload_metrics": reload_metrics,
        },
        "training_curve": {
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "initial_accuracy": accuracies[0],
            "final_accuracy": accuracies[-1],
            "min_loss": min(losses),
            "min_loss_epoch": int(training_rows[int(np.argmin(losses))]["epoch"]),
        },
        "round4_fold00_reference": round4_reference,
    }
    write_json(fold_root / "test_metrics.json", {"full": full_metrics, "noaug": noaug_metrics})
    write_json(fold_root / "config.json", {"fold": fold_id, "seed": seed, "protocol": PROTOCOL})
    write_json(fold_root / "run_metadata.json", metadata)
    print("FULL_TEST", json.dumps(full_metrics, sort_keys=True))
    print("NOAUG_TEST", json.dumps(noaug_metrics, sort_keys=True))
    print("RELOAD", json.dumps(metadata["checkpoint_reload"], sort_keys=True))
    print(f"FOLD_{fold_id:02d}_DONE", f"health={metadata['training_health']}")
    return metadata


def sample_sd(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def confusion_metrics_from_matrix(matrix: np.ndarray) -> dict[str, object]:
    tn, fp = int(matrix[0, 0]), int(matrix[0, 1])
    fn, tp = int(matrix[1, 0]), int(matrix[1, 1])
    total = tn + fp + fn + tp
    def class_metrics(tp_value: int, fp_value: int, fn_value: int) -> dict[str, float]:
        precision = tp_value / (tp_value + fp_value) if tp_value + fp_value else 0.0
        recall = tp_value / (tp_value + fn_value) if tp_value + fn_value else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}
    td = class_metrics(tn, fn, fp)
    asd = class_metrics(tp, fp, fn)
    return {
        "samples": total,
        "accuracy": (tn + tp) / total if total else 0.0,
        "confusion_matrix": matrix.tolist(),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "TD": td, "ASD": asd,
        "macro_f1": (td["f1"] + asd["f1"]) / 2,
    }


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def rank_values(values: list[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=float), kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def aggregate(output_root: Path, split_summaries: list[dict[str, object]]) -> dict[str, object]:
    records = []
    for fold_id in range(FOLD_COUNT):
        path = output_root / f"fold_{fold_id:02d}" / "run_metadata.json"
        if not path.exists():
            raise FileNotFoundError(f"missing completed fold: {path}")
        metadata = json.loads(path.read_text(encoding="utf-8"))
        full = metadata["full_test"]
        noaug = metadata["noaug_test"]
        duplicate = metadata["duplicate_analysis"]
        duplicated = duplicate["duplicated_in_train_test_subset"]
        nonduplicated = duplicate["nonduplicated_test_subset"]
        sentinel_count = len(metadata["sentinel_analysis"]["original_samples"])
        records.append(
            {
                "fold": fold_id,
                "seed": metadata["seed"],
                "full_test_loss": full["loss"],
                "full_test_accuracy": full["accuracy"],
                "macro_F1": full["macro_f1"],
                "TD_precision": full["TD"]["precision"],
                "TD_recall": full["TD"]["recall"],
                "TD_F1": full["TD"]["f1"],
                "ASD_precision": full["ASD"]["precision"],
                "ASD_recall": full["ASD"]["recall"],
                "ASD_F1": full["ASD"]["f1"],
                "TN": full["TN"], "FP": full["FP"], "FN": full["FN"], "TP": full["TP"],
                "NoAug_accuracy": noaug["accuracy"],
                "NoAug_macro_F1": noaug["macro_f1"],
                "duplicate_overlap_count": metadata["split"]["duplicate_hash_overlap"],
                "duplicated_subset_count": duplicated["count"],
                "duplicated_subset_accuracy": duplicated["accuracy"],
                "nonduplicated_subset_count": nonduplicated["count"],
                "nonduplicated_subset_accuracy": nonduplicated["accuracy"],
                "sentinel_test_count": sentinel_count,
                "training_health": metadata["training_health"],
                "reload_prediction_equality": metadata["checkpoint_reload"]["prediction_equality"],
                "reload_max_logit_difference": metadata["checkpoint_reload"]["max_logit_difference"],
            }
        )
    fields = list(records[0])
    with (output_root / "aggregate_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    accuracies = [float(row["full_test_accuracy"]) for row in records]
    macro_f1 = [float(row["macro_F1"]) for row in records]
    td_recall = [float(row["TD_recall"]) for row in records]
    asd_recall = [float(row["ASD_recall"]) for row in records]
    noaug_accuracy = [float(row["NoAug_accuracy"]) for row in records]
    pooled_matrix = np.sum(
        [np.asarray([[row["TN"], row["FP"]], [row["FN"], row["TP"]]], dtype=np.int64) for row in records],
        axis=0,
    )
    pooled = confusion_metrics_from_matrix(pooled_matrix)
    duplicate_count = sum(int(row["duplicated_subset_count"]) for row in records)
    nonduplicate_count = sum(int(row["nonduplicated_subset_count"]) for row in records)
    duplicate_correct = sum(
        int(round(float(row["duplicated_subset_accuracy"]) * int(row["duplicated_subset_count"])))
        for row in records
        if row["duplicated_subset_accuracy"] is not None
    )
    nonduplicate_correct = sum(
        int(round(float(row["nonduplicated_subset_accuracy"]) * int(row["nonduplicated_subset_count"])))
        for row in records
        if row["nonduplicated_subset_accuracy"] is not None
    )
    duplicated_accuracy = duplicate_correct / duplicate_count if duplicate_count else None
    nonduplicated_accuracy = nonduplicate_correct / nonduplicate_count if nonduplicate_count else None
    overlap = [float(row["duplicate_overlap_count"]) for row in records]
    aggregate_data = {
        "round": 5,
        "protocol": PROTOCOL,
        "fold_count": FOLD_COUNT,
        "rows": records,
        "full_test_accuracy": {
            "mean": float(np.mean(accuracies)),
            "sample_sd": sample_sd(accuracies),
            "median": float(np.median(accuracies)),
            "min": float(np.min(accuracies)),
            "max": float(np.max(accuracies)),
            "sd_formula": "sample standard deviation, ddof=1",
        },
        "macro_F1_mean_sd": {"mean": float(np.mean(macro_f1)), "sample_sd": sample_sd(macro_f1)},
        "TD_recall_mean_sd": {"mean": float(np.mean(td_recall)), "sample_sd": sample_sd(td_recall)},
        "ASD_recall_mean_sd": {"mean": float(np.mean(asd_recall)), "sample_sd": sample_sd(asd_recall)},
        "NoAug_accuracy_mean_sd": {"mean": float(np.mean(noaug_accuracy)), "sample_sd": sample_sd(noaug_accuracy)},
        "pooled_confusion": pooled,
        "duplicate_analysis": {
            "total_duplicated_test_evaluations": duplicate_count,
            "duplicated_accuracy": duplicated_accuracy,
            "total_nonduplicated_test_evaluations": nonduplicate_count,
            "nonduplicated_accuracy": nonduplicated_accuracy,
            "accuracy_gap_duplicated_minus_nonduplicated": (
                duplicated_accuracy - nonduplicated_accuracy
                if duplicated_accuracy is not None and nonduplicated_accuracy is not None else None
            ),
            "per_fold": [
                {
                    "fold": row["fold"],
                    "count_duplicated": row["duplicated_subset_count"],
                    "accuracy_duplicated": row["duplicated_subset_accuracy"],
                    "count_nonduplicated": row["nonduplicated_subset_count"],
                    "accuracy_nonduplicated": row["nonduplicated_subset_accuracy"],
                }
                for row in records
            ],
        },
        "duplicate_count_exploratory_correlation": {
            "pearson": pearson(overlap, accuracies),
            "spearman": pearson(rank_values(overlap).tolist(), rank_values(accuracies).tolist()),
            "n": FOLD_COUNT,
            "note": "exploratory only; n=10 and not causal",
        },
        "paper_comparison": {
            "paper_mean_accuracy": 0.9124,
            "paper_sample_sd": 0.0319,
            "ours_mean_accuracy": float(np.mean(accuracies)),
            "ours_sample_sd": sample_sd(accuracies),
            "absolute_mean_difference": abs(float(np.mean(accuracies)) - 0.9124),
            "interpretation": "cautious comparison; reconstructed folds/seeds/preprocessing may differ",
        },
        "sentinel_results": [
            json.loads((output_root / f"fold_{fold_id:02d}" / "sentinel_analysis.json").read_text(encoding="utf-8"))["original_samples"]
            for fold_id in range(FOLD_COUNT)
        ],
        "health": {
            "folds": {f"fold_{row['fold']:02d}": row["training_health"] for row in records},
            "failed_or_suspicious": [row["fold"] for row in records if row["training_health"] != "HEALTHY"],
            "all_reload_equal": all(row["reload_prediction_equality"] for row in records),
            "max_observed_logit_difference": max(float(row["reload_max_logit_difference"]) for row in records),
        },
        "fold_integrity": split_summaries,
        "interpretation": {
            "same_region_as_paper": "assess only after inspecting mean and uncertainty; not an exact reproduction claim",
            "duplicate_leakage": "performance on duplicated and nonduplicated samples is descriptive, not causal",
            "class_behavior": "compare TD and ASD recall means; do not infer from a single fold",
            "round_6_recommendation": "proceed to leakage-safe evaluation only after reviewing this report",
        },
    }
    write_json(output_root / "aggregate_metrics.json", aggregate_data)
    print("ROUND5_AGGREGATE", json.dumps(aggregate_data["full_test_accuracy"], sort_keys=True))
    print("ROUND5_POOLED", json.dumps(pooled, sort_keys=True))
    return aggregate_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fold", default="all", help="0-9 or all; all runs sequentially")
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--resume", action="store_true", help="reuse healthy completed folds")
    parser.add_argument("--overwrite", action="store_true", help="overwrite incomplete fold artifacts")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    if args.seed != BASE_SEED:
        raise ValueError(f"Round 5 seed is frozen at {BASE_SEED}; do not change it")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)
    if str(args.fold).lower() == "all":
        fold_ids = list(range(FOLD_COUNT))
    else:
        fold_ids = [int(args.fold)]
        if fold_ids[0] not in range(FOLD_COUNT):
            raise ValueError("--fold must be all or an integer from 0 to 9")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = read_manifest()
    split_summaries, sentinel_map = validate_all_folds(rows)
    write_json(output_root / "split_verification.json", split_summaries)
    write_json(output_root / "sentinel_placement.json", sentinel_map)
    print("ALL_FOLDS_VERIFIED", json.dumps(split_summaries, sort_keys=True))
    print("SENTINEL_PLACEMENT", json.dumps(sentinel_map, sort_keys=True))

    # Snapshot the dirty repository before the first fold.  The diff checksum
    # is checked after every fold so the code state cannot drift mid-CV.
    seed_state = set_seed(BASE_SEED)
    code_state = snapshot_code_state(output_root, device)
    diff_path = output_root / "code_state" / "git_diff.patch"
    initial_diff_sha256 = sha256_file(diff_path)
    config = frozen_config(seed_state, code_state, [BASE_SEED + fold for fold in range(FOLD_COUNT)])
    write_immutable_config(output_root / "config.json", config)
    write_json(output_root / "run_metadata.json", {
        "round": 5,
        "protocol": PROTOCOL,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if torch.cuda.is_available() else None,
        "code_state": code_state,
        "initial_diff_sha256": initial_diff_sha256,
        "fold_order": fold_ids,
        "test_evaluation_during_training": False,
    })

    for fold_id in fold_ids:
        fold_metadata = run_fold(
            fold_id=fold_id,
            seed=BASE_SEED + fold_id,
            rows=rows,
            output_root=output_root,
            device=device,
            code_state=code_state,
            split_summary=split_summaries[fold_id],
            resume=args.resume,
            overwrite=args.overwrite,
        )
        current_diff_sha256 = sha256_file(diff_path)
        if current_diff_sha256 != initial_diff_sha256:
            raise RuntimeError("git diff changed between folds; stop the experiment")
        print(f"FOLD_{fold_id:02d}_HEALTH", fold_metadata["training_health"])

    if len(fold_ids) == FOLD_COUNT:
        aggregate(output_root, split_summaries)
        print("ROUND5_DONE", f"experiment_root={output_root}")
    else:
        completed = all((output_root / f"fold_{fold:02d}" / "run_metadata.json").exists() for fold in range(FOLD_COUNT))
        if completed:
            aggregate(output_root, split_summaries)
            print("ROUND5_DONE", f"experiment_root={output_root}")
        else:
            print("ROUND5_PARTIAL", f"completed_fold={fold_ids[0]:02d}", f"experiment_root={output_root}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train exactly one Round 4 Dryad/MSG3D fold.

This runner is deliberately separate from the legacy multimodal training
entrypoint.  It trains only random_reproduction_like/fold_00, evaluates the
test split after epoch 30, and never uses test metrics for model selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from prepare_dryad_msg3d import SerializedFoldDataset  # noqa: E402


PROTOCOL = "random_reproduction_like"
FOLD_ID = 0
ROUND4_SEED = 2026091704
EPOCHS = 30
BATCH_SIZE = 32
FORWARD_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
BASE_LR = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NESTEROV = True
MILESTONES = [10, 20, 30, 40]
GAMMA = 0.1
ARTIFACT_ROOT = ROOT / "artifacts" / "dryad_msg3d"
DEFAULT_EXPERIMENT_ROOT = ARTIFACT_ROOT / "experiments" / "round4_fold00"
CHECKPOINT = ROOT / "autism" / "msg3d_pretrained_weights" / "ntu120-xsub-joint.pt"
EARLY_PARAMETER = "gcn3d1.gcn3d.0.gcn3d.1.mlp.layers.0.weight"
LATE_PARAMETER = "gcn3d3.gcn3d.0.gcn3d.1.mlp.layers.0.weight"


def json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=json_default)
        + "\n",
        encoding="utf-8",
    )


def set_seed(seed: int) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_seeded = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cuda_seeded = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    deterministic_algorithms = False
    deterministic_error = None
    try:
        torch.use_deterministic_algorithms(True)
        deterministic_algorithms = True
    except RuntimeError as exc:
        deterministic_error = str(exc)
    return {
        "seed": seed,
        "python_random": True,
        "numpy": True,
        "torch_cpu": True,
        "torch_cuda": cuda_seeded,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "torch_deterministic_algorithms": deterministic_algorithms,
        "torch_deterministic_error": deterministic_error,
    }


def git_output(args: list[str]) -> str:
    result = subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True)
    return result.stdout


def snapshot_code_state(experiment_root: Path, device: torch.device) -> dict[str, object]:
    code_root = experiment_root / "code_state"
    code_root.mkdir(parents=True, exist_ok=True)
    (code_root / "git_commit.txt").write_text(git_output(["git", "rev-parse", "HEAD"]), encoding="utf-8")
    (code_root / "git_status.txt").write_text(git_output(["git", "status", "--short"]), encoding="utf-8")
    (code_root / "git_diff_stat.txt").write_text(git_output(["git", "diff", "--stat"]), encoding="utf-8")
    diff = subprocess.run(
        ["git", "diff", "--binary"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    (code_root / "git_diff.patch").write_bytes(diff)
    environment_lines = [
        f"python={platform.python_version()}",
        f"python_executable={sys.executable}",
        f"torch={torch.__version__}",
        f"numpy={np.__version__}",
        f"device={device}",
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
        f"cuda_available={torch.cuda.is_available()}",
        f"cuda_version={torch.version.cuda}",
        f"cuda_device_count={torch.cuda.device_count()}",
        f"cudnn_version={torch.backends.cudnn.version()}",
        f"torch_num_threads={torch.get_num_threads()}",
    ]
    if torch.cuda.is_available():
        environment_lines.extend(
            [
                f"gpu_name={torch.cuda.get_device_name(device)}",
                f"gpu_capability={torch.cuda.get_device_capability(device)}",
            ]
        )
    try:
        nvidia_smi = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=False
        )
        environment_lines.append("nvidia_smi_exit=" + str(nvidia_smi.returncode))
        environment_lines.append(nvidia_smi.stdout)
        environment_lines.append(nvidia_smi.stderr)
    except OSError as exc:
        environment_lines.append(f"nvidia_smi_error={exc}")
    (code_root / "environment.txt").write_text("\n".join(environment_lines) + "\n", encoding="utf-8")
    return {
        "git_commit": (code_root / "git_commit.txt").read_text(encoding="utf-8").strip(),
        "git_dirty": bool((code_root / "git_status.txt").read_text(encoding="utf-8").strip()),
        "code_state_dir": str(code_root),
    }


def read_manifest() -> list[dict[str, str]]:
    with (ARTIFACT_ROOT / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_fold() -> dict[str, object]:
    path = ARTIFACT_ROOT / "folds" / PROTOCOL / f"fold_{FOLD_ID:02d}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def verify_fold(rows: list[dict[str, str]], fold: dict[str, object]) -> dict[str, object]:
    if fold["protocol"] != PROTOCOL or int(fold["fold_id"]) != FOLD_ID:
        raise AssertionError("wrong protocol/fold artifact")
    if fold["train_sample_count"] != 720 or fold["test_sample_count"] != 80:
        raise AssertionError("fold_00 sample count mismatch")
    if fold["train_subject_count"] != 90 or fold["test_subject_count"] != 10:
        raise AssertionError("fold_00 subject count mismatch")
    train_ids = set(fold["train_sample_ids"])
    test_ids = set(fold["test_sample_ids"])
    train_subjects = set(fold["train_subject_ids"])
    test_subjects = set(fold["test_subject_ids"])
    if train_ids & test_ids or train_subjects & test_subjects:
        raise AssertionError("subject/sample leakage in fold_00")
    by_sample = {row["sample_id"]: row for row in rows}
    if train_ids | test_ids != set(by_sample):
        raise AssertionError("fold sample IDs do not cover manifest")
    for subject_id in train_subjects | test_subjects:
        partitions = {
            "train" if row["sample_id"] in train_ids else "test"
            for row in rows
            if row["subject_id"] == subject_id
        }
        if len(partitions) != 1:
            raise AssertionError(f"augmentation partition leakage for {subject_id}")
    expected_counts = {
        "train_ASD": 360,
        "train_TD": 360,
        "test_ASD": 40,
        "test_TD": 40,
    }
    actual_counts = {
        "train_ASD": sum(by_sample[sample]["class_name"] == "ASD" for sample in train_ids),
        "train_TD": sum(by_sample[sample]["class_name"] == "TD" for sample in train_ids),
        "test_ASD": sum(by_sample[sample]["class_name"] == "ASD" for sample in test_ids),
        "test_TD": sum(by_sample[sample]["class_name"] == "TD" for sample in test_ids),
    }
    if actual_counts != expected_counts:
        raise AssertionError(f"class counts mismatch: {actual_counts}")
    if fold["duplicate_leakage_count"] != 24:
        raise AssertionError(f"expected 24 duplicate hash overlaps, got {fold['duplicate_leakage_count']}")
    return {
        "train_sample_count": len(train_ids),
        "test_sample_count": len(test_ids),
        "train_subject_count": len(train_subjects),
        "test_subject_count": len(test_subjects),
        **actual_counts,
        "duplicate_hash_overlap": int(fold["duplicate_leakage_count"]),
    }


def sentinel_exposure(rows: list[dict[str, str]], fold: dict[str, object]) -> dict[str, object]:
    train_ids = set(fold["train_sample_ids"])
    test_ids = set(fold["test_sample_ids"])
    affected = [row for row in rows if int(row["sentinel_count"]) > 0]
    by_subject: dict[str, dict[str, object]] = {}
    for row in affected:
        subject = row["subject_id"]
        partition = "train" if row["sample_id"] in train_ids else "test" if row["sample_id"] in test_ids else "unknown"
        entry = by_subject.setdefault(
            subject,
            {
                "subject_id": subject,
                "partition": partition,
                "original_sample_present": False,
                "affected_augmentation_samples": [],
                "total_sentinel_count": 0,
            },
        )
        if entry["partition"] != partition:
            raise AssertionError("sentinel subject crosses fold partition")
        entry["total_sentinel_count"] += int(row["sentinel_count"])
        if row["is_original"] == "1":
            entry["original_sample_present"] = True
        else:
            entry["affected_augmentation_samples"].append(row["sample_id"])
    result = {
        "subjects": [by_subject[key] for key in sorted(by_subject)],
        "train_subjects": sorted(
            {entry["subject_id"] for entry in by_subject.values() if entry["partition"] == "train"}
        ),
        "test_subjects": sorted(
            {entry["subject_id"] for entry in by_subject.values() if entry["partition"] == "test"}
        ),
        "train_sentinel_count": sum(
            int(row["sentinel_count"]) for row in affected if row["sample_id"] in train_ids
        ),
        "test_sentinel_count": sum(
            int(row["sentinel_count"]) for row in affected if row["sample_id"] in test_ids
        ),
    }
    return result


class TorchSerializedDataset(Dataset):
    def __init__(self, serialized: SerializedFoldDataset):
        self.serialized = serialized

    def __len__(self):
        return len(self.serialized)

    def __getitem__(self, index):
        data, label, source_path = self.serialized[index]
        return torch.from_numpy(data), int(label), source_path


def make_loader(data_path: Path, labels_path: Path, batch_size: int, shuffle: bool, seed: int):
    dataset = TorchSerializedDataset(SerializedFoldDataset(data_path, labels_path))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def import_plain_model():
    autism_root = ROOT / "autism"
    if str(autism_root) not in sys.path:
        sys.path.insert(0, str(autism_root))
    from plain_msg3d import build_plain_msg3d, load_ntu120_backbone

    return build_plain_msg3d, load_ntu120_backbone


def build_loaded_model(device: torch.device):
    build_plain_msg3d, load_ntu120_backbone = import_plain_model()
    model = build_plain_msg3d(num_class=2, num_point=25, num_person=2).to(device)
    load_info = load_ntu120_backbone(model, CHECKPOINT, map_location=device)
    if model.dream is not False or model.use_angle is not False:
        raise AssertionError("plain model is not angle-free")
    if tuple(model.fc.weight.shape) != (2, 384):
        raise AssertionError(f"unexpected FC shape: {tuple(model.fc.weight.shape)}")
    if load_info["ignored_keys"] != ["fc.bias", "fc.weight"]:
        raise AssertionError(load_info)
    if load_info["missing_keys"] != ["fc.weight", "fc.bias"] or load_info["unexpected_keys"]:
        raise AssertionError(load_info)
    return model, load_info


def parameter_report(model: nn.Module) -> dict[str, object]:
    parameters = dict(model.named_parameters())
    for name in ("fc.weight", EARLY_PARAMETER, LATE_PARAMETER):
        if name not in parameters:
            raise AssertionError(f"required diagnostic parameter not found: {name}")
    frozen = [name for name, parameter in parameters.items() if not parameter.requires_grad]
    trainable = [name for name, parameter in parameters.items() if parameter.requires_grad]
    if not parameters["fc.weight"].requires_grad:
        raise AssertionError("final FC is frozen")
    return {
        "total": sum(parameter.numel() for parameter in parameters.values()),
        "trainable": sum(parameter.numel() for parameter in parameters.values() if parameter.requires_grad),
        "frozen": sum(parameter.numel() for parameter in parameters.values() if not parameter.requires_grad),
        "frozen_names": frozen,
        "trainable_parameter_count": len(trainable),
        "full_finetuning": not frozen,
        "diagnostic_parameters": {
            "fc": "fc.weight",
            "early": EARLY_PARAMETER,
            "late": LATE_PARAMETER,
        },
    }


def gradient_report(model: nn.Module, names: dict[str, str]) -> dict[str, object]:
    parameters = dict(model.named_parameters())
    result: dict[str, object] = {}
    for label, name in names.items():
        gradient = parameters[name].grad
        result[label] = {
            "name": name,
            "exists": gradient is not None,
            "finite": bool(gradient is not None and torch.isfinite(gradient).all()),
            "norm": float(torch.linalg.vector_norm(gradient).item()) if gradient is not None else None,
        }
    return result


def check_all_gradients(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is None:
            raise RuntimeError(f"missing gradient: {name}")
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"non-finite gradient: {name}")


def run_batch_backward(
    model: nn.Module,
    optimizer: SGD,
    criterion: nn.Module,
    batch,
    device: torch.device,
    do_step: bool,
) -> dict[str, object]:
    data, labels, _paths = batch
    data = data.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True).long()
    if not set(labels.detach().cpu().tolist()).issubset({0, 1}):
        raise ValueError("invalid label")
    if data.shape[0] % FORWARD_BATCH_SIZE != 0:
        raise ValueError("batch size is not divisible by forward batch size")
    splits = data.shape[0] // FORWARD_BATCH_SIZE
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    prediction_counts = {0: 0, 1: 0}
    for split_index in range(splits):
        left = split_index * FORWARD_BATCH_SIZE
        right = left + FORWARD_BATCH_SIZE
        output = model(data[left:right])
        if tuple(output.shape) != (right - left, 2):
            raise RuntimeError(f"invalid output shape: {tuple(output.shape)}")
        batch_loss = criterion(output, labels[left:right])
        if not torch.isfinite(batch_loss):
            raise FloatingPointError("non-finite loss")
        (batch_loss / splits).backward()
        prediction = output.detach().argmax(dim=1)
        prediction_counts[0] += int((prediction == 0).sum())
        prediction_counts[1] += int((prediction == 1).sum())
        total_correct += int((prediction == labels[left:right]).sum())
        total_samples += right - left
        total_loss += float(batch_loss.item()) * (right - left)
    check_all_gradients(model)
    if do_step:
        optimizer.step()
        for name, parameter in model.named_parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError(f"non-finite parameter after optimizer step: {name}")
    return {
        "loss_sum": total_loss,
        "samples": total_samples,
        "correct": total_correct,
        "prediction_counts": prediction_counts,
    }


def run_gradient_diagnostic(device: torch.device, batch) -> dict[str, object]:
    model, load_info = build_loaded_model(device)
    model.train()
    optimizer = SGD(
        model.parameters(), lr=BASE_LR, momentum=MOMENTUM,
        nesterov=NESTEROV, weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()
    stats = run_batch_backward(model, optimizer, criterion, batch, device, do_step=False)
    gradients = gradient_report(
        model,
        {"fc": "fc.weight", "early": EARLY_PARAMETER, "late": LATE_PARAMETER},
    )
    optimizer.zero_grad(set_to_none=True)
    result = {
        "loss": stats["loss_sum"] / stats["samples"],
        "gradients": gradients,
        "all_gradients_finite": all(item["finite"] for item in gradients.values()),
        "checkpoint_load": load_info,
    }
    if not result["all_gradients_finite"] or not all(item["exists"] for item in gradients.values()):
        raise AssertionError(result)
    return result


def run_optimizer_diagnostic(device: torch.device, batch) -> dict[str, object]:
    model, load_info = build_loaded_model(device)
    model.train()
    optimizer = SGD(
        model.parameters(), lr=BASE_LR, momentum=MOMENTUM,
        nesterov=NESTEROV, weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()
    before_fc = model.fc.weight.detach().clone()
    before_backbone = dict(model.named_parameters())[EARLY_PARAMETER].detach().clone()
    stats = run_batch_backward(model, optimizer, criterion, batch, device, do_step=True)
    after_fc = model.fc.weight.detach()
    after_backbone = dict(model.named_parameters())[EARLY_PARAMETER].detach()
    fc_delta = float((after_fc - before_fc).abs().max())
    backbone_delta = float((after_backbone - before_backbone).abs().max())
    result = {
        "loss": stats["loss_sum"] / stats["samples"],
        "fc_changed": fc_delta > 0.0,
        "backbone_changed": backbone_delta > 0.0,
        "fc_max_abs_delta": fc_delta,
        "backbone_max_abs_delta": backbone_delta,
        "checkpoint_load": load_info,
    }
    if not result["fc_changed"] or not result["backbone_changed"]:
        raise AssertionError(result)
    del model, optimizer
    return result


def train_epoch(model, optimizer, criterion, loader, device) -> dict[str, object]:
    model.train()
    loss_sum = 0.0
    correct = 0
    samples = 0
    prediction_counts = {0: 0, 1: 0}
    for batch in loader:
        stats = run_batch_backward(model, optimizer, criterion, batch, device, do_step=True)
        loss_sum += float(stats["loss_sum"])
        correct += int(stats["correct"])
        samples += int(stats["samples"])
        prediction_counts[0] += int(stats["prediction_counts"][0])
        prediction_counts[1] += int(stats["prediction_counts"][1])
    return {
        "loss": loss_sum / samples,
        "accuracy": correct / samples,
        "samples": samples,
        "predicted_td": prediction_counts[0],
        "predicted_asd": prediction_counts[1],
    }


def confusion_metrics(labels: np.ndarray, predictions: np.ndarray, loss: float) -> dict[str, object]:
    confusion = np.zeros((2, 2), dtype=np.int64)
    for true, predicted in zip(labels, predictions):
        confusion[int(true), int(predicted)] += 1
    per_class = {}
    for class_id, class_name in ((0, "TD"), (1, "ASD")):
        tp = int(confusion[class_id, class_id])
        fp = int(confusion[:, class_id].sum() - tp)
        fn = int(confusion[class_id, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[class_name] = {"precision": precision, "recall": recall, "f1": f1}
    return {
        "samples": int(labels.size),
        "loss": float(loss),
        "accuracy": float((labels == predictions).mean()),
        "confusion_matrix": confusion.tolist(),
        "TN": int(confusion[0, 0]),
        "FP": int(confusion[0, 1]),
        "FN": int(confusion[1, 0]),
        "TP": int(confusion[1, 1]),
        "TD": per_class["TD"],
        "ASD": per_class["ASD"],
        "macro_f1": (per_class["TD"]["f1"] + per_class["ASD"]["f1"]) / 2,
    }


def evaluate(model, loader, device, rows_by_path: dict[str, dict[str, str]]):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    all_labels = []
    all_predictions = []
    all_logits = []
    sample_results = []
    loss_sum = 0.0
    with torch.no_grad():
        for data, labels, paths in loader:
            data = data.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True).long()
            logits = model(data)
            if tuple(logits.shape) != (data.shape[0], 2):
                raise RuntimeError(f"invalid eval output shape: {tuple(logits.shape)}")
            if not torch.isfinite(logits).all():
                raise FloatingPointError("non-finite eval logits")
            loss_sum += float(criterion(logits, labels_device).item())
            predictions = logits.argmax(dim=1)
            probabilities = torch.softmax(logits, dim=1)
            paths_list = list(paths)
            for index, source_path in enumerate(paths_list):
                row = rows_by_path[source_path]
                sample_results.append(
                    {
                        "sample_id": row["sample_id"],
                        "source_path": source_path,
                        "subject_id": row["subject_id"],
                        "true_label": int(labels[index]),
                        "predicted_label": int(predictions[index]),
                        "logits": logits[index].detach().cpu().tolist(),
                        "probabilities": probabilities[index].detach().cpu().tolist(),
                        "sentinel_count": int(row["sentinel_count"]),
                        "numeric_hash": row["numeric_hash"],
                    }
                )
            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_logits.append(logits.cpu().numpy())
    labels_np = np.asarray(all_labels, dtype=np.int64)
    predictions_np = np.asarray(all_predictions, dtype=np.int64)
    logits_np = np.concatenate(all_logits, axis=0)
    metrics = confusion_metrics(labels_np, predictions_np, loss_sum / len(labels_np))
    return metrics, sample_results, logits_np, predictions_np


def save_confusion_csv(path: Path, confusion: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["", "predicted_TD", "predicted_ASD"])
        writer.writerow(["actual_TD", confusion[0][0], confusion[0][1]])
        writer.writerow(["actual_ASD", confusion[1][0], confusion[1][1]])


def evaluate_subset_accuracy(results: list[dict[str, object]]) -> dict[str, object]:
    if not results:
        return {"count": 0, "accuracy": None, "sample_ids": []}
    correct = sum(row["true_label"] == row["predicted_label"] for row in results)
    return {
        "count": len(results),
        "accuracy": correct / len(results),
        "sample_ids": [row["sample_id"] for row in results],
    }


def artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--protocol", default=PROTOCOL)
    parser.add_argument("--fold", type=int, default=FOLD_ID)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=ROUND4_SEED)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.protocol != PROTOCOL or args.fold != FOLD_ID:
        raise ValueError("Round 4 runner is restricted to random_reproduction_like/fold_00")
    if args.epochs != EPOCHS:
        raise ValueError("Round 4 requires exactly 30 epochs")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)
    experiment_root = args.experiment_root.resolve()
    if experiment_root.exists() and any(experiment_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"experiment already exists: {experiment_root}; use --overwrite")
    experiment_root.mkdir(parents=True, exist_ok=True)

    seed_state = set_seed(args.seed)
    code_state = snapshot_code_state(experiment_root, device)
    rows = read_manifest()
    fold = read_fold()
    split_summary = verify_fold(rows, fold)
    exposure = sentinel_exposure(rows, fold)
    write_json(experiment_root / "split_verification.json", split_summary)
    write_json(experiment_root / "sentinel_exposure.json", exposure)
    print("SPLIT", json.dumps(split_summary, sort_keys=True))
    print("SENTINEL", json.dumps(exposure, sort_keys=True))

    config = {
        "round": 4,
        "protocol": PROTOCOL,
        "fold": FOLD_ID,
        "seed": args.seed,
        "device": str(device),
        "epochs": EPOCHS,
        "optimizer": "SGD",
        "learning_rate": BASE_LR,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "nesterov": NESTEROV,
        "batch_size": BATCH_SIZE,
        "forward_batch_size": FORWARD_BATCH_SIZE,
        "test_batch_size": TEST_BATCH_SIZE,
        "scheduler": "MultiStepLR",
        "milestones": MILESTONES,
        "gamma": GAMMA,
        "scheduler_step_timing": "end_of_epoch",
        "loss": "CrossEntropyLoss",
        "architecture": "plain MSG3D",
        "dream": False,
        "use_angle": False,
        "skepxel": False,
        "vit": False,
        "auxiliary_loss": False,
        "checkpoint_source": str(CHECKPOINT),
        "test_evaluation_during_training": False,
        "validation": False,
        "data_contract": "DATA_CONTRACT_v1",
        "class_mapping": {"TD": 0, "ASD": 1},
        "seed_state": seed_state,
        "code_state": code_state,
        "torch_num_threads": torch.get_num_threads(),
    }
    write_json(experiment_root / "config.json", config)

    loader_root = ARTIFACT_ROOT / "loader" / PROTOCOL
    train_loader = make_loader(
        loader_root / "train_data_fold_00.npy",
        loader_root / "train_labels_fold_00.pkl",
        BATCH_SIZE,
        shuffle=False,
        seed=args.seed + 1,
    )
    diagnostic_batch = next(iter(train_loader))
    parameter_model, load_info = build_loaded_model(device)
    parameters = parameter_report(parameter_model)
    write_json(experiment_root / "parameter_report.json", {**parameters, "checkpoint_load": load_info})
    del parameter_model
    gradient = run_gradient_diagnostic(device, diagnostic_batch)
    optimizer_diag = run_optimizer_diagnostic(device, diagnostic_batch)
    write_json(experiment_root / "gradient_diagnostic.json", gradient)
    write_json(experiment_root / "optimizer_step_diagnostic.json", optimizer_diag)
    print("GRADIENT", json.dumps(gradient, sort_keys=True))
    print("OPTIMIZER_STEP", json.dumps(optimizer_diag, sort_keys=True))

    model, load_info = build_loaded_model(device)
    optimizer = SGD(
        model.parameters(), lr=BASE_LR, momentum=MOMENTUM,
        nesterov=NESTEROV, weight_decay=WEIGHT_DECAY,
    )
    scheduler = MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA)
    criterion = nn.CrossEntropyLoss()
    checkpoint_dir = experiment_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "epoch_00.pt")
    log_path = experiment_root / "training_log.csv"
    with log_path.open("w", encoding="utf-8", newline="") as log_handle:
        fieldnames = [
            "epoch", "learning_rate", "learning_rate_after_scheduler", "train_loss",
            "train_accuracy", "samples", "predicted_td", "predicted_asd", "elapsed_seconds",
        ]
        writer = csv.DictWriter(log_handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.perf_counter()
            epoch_loader = make_loader(
                loader_root / "train_data_fold_00.npy",
                loader_root / "train_labels_fold_00.pkl",
                BATCH_SIZE,
                shuffle=True,
                seed=args.seed + epoch,
            )
            learning_rate = float(optimizer.param_groups[0]["lr"])
            metrics = train_epoch(model, optimizer, criterion, epoch_loader, device)
            scheduler.step()
            learning_rate_after = float(optimizer.param_groups[0]["lr"])
            row = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "learning_rate_after_scheduler": learning_rate_after,
                "train_loss": metrics["loss"],
                "train_accuracy": metrics["accuracy"],
                "samples": metrics["samples"],
                "predicted_td": metrics["predicted_td"],
                "predicted_asd": metrics["predicted_asd"],
                "elapsed_seconds": time.perf_counter() - epoch_start,
            }
            writer.writerow(row)
            log_handle.flush()
            print("TRAIN", json.dumps(row, sort_keys=True))
            if epoch in {10, 20, 30}:
                torch.save(model.state_dict(), checkpoint_dir / f"epoch_{epoch:02d}.pt")

    final_model_path = experiment_root / "final_model.pt"
    optimizer_path = experiment_root / "optimizer_state.pt"
    scheduler_path = experiment_root / "scheduler_state.pt"
    torch.save(model.state_dict(), final_model_path)
    torch.save(optimizer.state_dict(), optimizer_path)
    torch.save(scheduler.state_dict(), scheduler_path)

    by_path = {row["source_path"]: row for row in rows}
    test_loader = make_loader(
        loader_root / "test_data_fold_00.npy",
        loader_root / "test_labels_fold_00.pkl",
        TEST_BATCH_SIZE,
        shuffle=False,
        seed=args.seed + 100,
    )
    full_metrics, full_results, full_logits, full_predictions = evaluate(model, test_loader, device, by_path)
    write_json(experiment_root / "test_predictions.json", full_results)
    save_confusion_csv(experiment_root / "confusion_matrix.csv", full_metrics["confusion_matrix"])

    noaug_root = ARTIFACT_ROOT / "noaug" / PROTOCOL
    noaug_loader = make_loader(
        noaug_root / "test_data_NoAug_fold_00.npy",
        noaug_root / "test_labels_NoAug_fold_00.pkl",
        TEST_BATCH_SIZE,
        shuffle=False,
        seed=args.seed + 101,
    )
    noaug_metrics, noaug_results, _noaug_logits, _noaug_predictions = evaluate(
        model, noaug_loader, device, by_path
    )

    train_hashes = {
        rows[index]["numeric_hash"] for index in fold["train_indices"] if rows[index]["numeric_hash"]
    }
    duplicated_test = [row for row in full_results if row["numeric_hash"] in train_hashes]
    nonduplicated_test = [row for row in full_results if row["numeric_hash"] not in train_hashes]
    duplicate_analysis = {
        "train_test_numeric_hash_overlap_count": fold["duplicate_leakage_count"],
        "duplicated_in_train_test_subset": evaluate_subset_accuracy(duplicated_test),
        "nonduplicated_test_subset": evaluate_subset_accuracy(nonduplicated_test),
    }
    sentinel_test = [row for row in full_results if row["sentinel_count"] > 0]
    nonsentinel_test = [row for row in full_results if row["sentinel_count"] == 0]
    sentinel_analysis = {
        "sentinel_test_samples": sentinel_test,
        "sentinel_subset": evaluate_subset_accuracy(sentinel_test),
        "nonsentinel_subset": evaluate_subset_accuracy(nonsentinel_test),
    }
    write_json(experiment_root / "duplicate_leakage_analysis.json", duplicate_analysis)
    write_json(experiment_root / "sentinel_analysis.json", sentinel_analysis)

    reload_model, reload_load_info = build_loaded_model(device)
    reload_state = torch.load(final_model_path, map_location=device)
    reload_model.load_state_dict(reload_state)
    reload_metrics, reload_results, reload_logits, reload_predictions = evaluate(
        reload_model, test_loader, device, by_path
    )
    max_logit_difference = float(np.max(np.abs(full_logits - reload_logits)))
    prediction_equality = bool(np.array_equal(full_predictions, reload_predictions))
    if not prediction_equality or max_logit_difference > 0.0:
        raise AssertionError(
            f"checkpoint reload mismatch: equal={prediction_equality}, max_diff={max_logit_difference}"
        )

    with log_path.open(newline="", encoding="utf-8") as handle:
        training_rows = list(csv.DictReader(handle))
    losses = [float(row["train_loss"]) for row in training_rows]
    accuracies = [float(row["train_accuracy"]) for row in training_rows]
    learning_rates = [float(row["learning_rate"]) for row in training_rows]
    healthy = (
        all(math.isfinite(value) for value in losses + accuracies)
        and abs(losses[-1] - losses[0]) > 1e-8
        and all(int(row["predicted_td"]) > 0 and int(row["predicted_asd"]) > 0 for row in training_rows[-3:])
    )
    run_metadata = {
        "round": 4,
        "protocol": PROTOCOL,
        "fold": FOLD_ID,
        "seed": args.seed,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(device) if torch.cuda.is_available() else None,
        "git": code_state,
        "split": split_summary,
        "sentinel_exposure": exposure,
        "model": {
            "architecture": "plain MSG3D",
            "num_class": 2,
            "dream": False,
            "use_angle": False,
            "skepxel": False,
            "vit": False,
            "auxiliary_loss": False,
            "checkpoint_load": load_info,
            "parameter_report": parameters,
        },
        "hyperparameters": config,
        "gradient_diagnostic": gradient,
        "optimizer_step_diagnostic": optimizer_diag,
        "training_health": "HEALTHY" if healthy else "SUSPICIOUS",
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
        },
        "training_curve": {
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "initial_accuracy": accuracies[0],
            "final_accuracy": accuracies[-1],
            "min_loss": min(losses),
            "min_loss_epoch": int(training_rows[int(np.argmin(losses))]["epoch"]),
            "learning_rates": learning_rates,
        },
        "paper_random_msg3d_context": "91.24 +/- 3.19 is a 10-fold paper mean and is not directly comparable to one reconstructed fold",
    }
    write_json(experiment_root / "test_metrics.json", {"full": full_metrics, "noaug": noaug_metrics})
    write_json(experiment_root / "run_metadata.json", run_metadata)
    print("FULL_TEST", json.dumps(full_metrics, sort_keys=True))
    print("NOAUG_TEST", json.dumps(noaug_metrics, sort_keys=True))
    print("RELOAD", json.dumps(run_metadata["checkpoint_reload"], sort_keys=True))
    print("ROUND4_DONE", f"experiment_root={experiment_root}", f"health={run_metadata['training_health']}")


if __name__ == "__main__":
    main()

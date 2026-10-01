"""Training/evaluation loops shared by CLI modules and the smoke test."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data.collate import skeleton_collate
from .data.dataset import SkeletonClipDataset, SkeletonWindowDataset, read_manifest
from .data.sequence import SequenceBuilder
from .metrics import classification_metrics
from .models.bilstm_attention import (
    BiLSTMAttentionClassifier,
    PaperBiLSTMAttentionClassifier,
    KinematicBiLSTMAttentionClassifier,
)
from .utils import assert_finite


def class_names_from_records(records: list[dict[str, Any]]) -> list[str]:
    mapping = {int(row["activity_id"]): str(row["activity"]) for row in records}
    if not mapping:
        raise ValueError("Cannot infer classes from an empty manifest")
    return [mapping[index] for index in range(max(mapping) + 1)]


def make_dataset(
    records: list[dict[str, Any]], config: dict[str, Any], return_debug: bool = False,
    train: bool = False,
) -> SkeletonClipDataset | SkeletonWindowDataset:
    data_config = dict(config.get("data", {}))
    data_config["project_root"] = config.get("_project_root", Path.cwd())
    data_config["seed"] = config.get("seed", 42)
    if "tracking" in config:
        data_config["tracking"] = dict(config["tracking"])
    if "augmentation" in config:
        data_config["augmentation"] = dict(config["augmentation"])
    builder = SequenceBuilder(dict(config.get("sequence", {})))
    dataset_type = SkeletonWindowDataset if builder.is_multi_window else SkeletonClipDataset
    return dataset_type(
        records,
        data_config=data_config,
        sequence_config=dict(config.get("sequence", {})),
        return_debug=return_debug,
        train=train,
    )


def make_loader(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    train: bool,
    max_samples: int | None = None,
    batch_size: int | None = None,
) -> DataLoader:
    selected = records[:max_samples] if max_samples else records
    dataset = make_dataset(selected, config, train=train)
    training_config = config.get("training", {})
    generator = torch.Generator()
    generator.manual_seed(int(config.get("seed", 42)))
    sampler = None
    shuffle = train
    sampler_name = str(training_config.get("sampler", "all_sequences"))
    if train and sampler_name == "clip_balanced":
        if not isinstance(dataset, SkeletonWindowDataset):
            raise ValueError("clip_balanced sampler is only meaningful for multiple-window modes")
        counts = Counter(str(row["source_clip_id"]) for row in dataset.windows)
        weights = torch.tensor([1.0 / counts[str(row["source_clip_id"])] for row in dataset.windows],
                               dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True, generator=generator)
        shuffle = False
    elif sampler_name != "all_sequences":
        raise ValueError("training.sampler must be all_sequences or clip_balanced")
    return DataLoader(
        dataset,
        batch_size=batch_size or int(training_config.get("batch_size", 32)),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(training_config.get("num_workers", 0)),
        collate_fn=skeleton_collate,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def build_model(config: dict[str, Any], num_classes: int) -> nn.Module:
    model_config = config.get("model", {})
    if model_config.get("type") == "kinematic_bilstm_attention":
        bilstm_config = dict(model_config.get("bilstm", {}))
        attention_config = dict(model_config.get("attention", {}))
        projection_config = dict(model_config.get("projection", {}))
        projection_size = int(projection_config.get("out_features", 64))
        hidden_size = int(bilstm_config.get("hidden_size", 128))
        bidirectional = bool(bilstm_config.get("bidirectional", True))
        return KinematicBiLSTMAttentionClassifier(
            input_size=50,
            hidden_size=hidden_size,
            num_layers=int(bilstm_config.get("num_layers", 2)),
            num_classes=num_classes,
            attention_heads=int(attention_config.get("num_heads", 4)),
            dropout=float(model_config.get("dropout", 0.3)),
            bidirectional=bidirectional,
            lstm_internal_dropout=float(model_config.get("lstm_internal_dropout", 0.0)),
            attention_dropout=float(attention_config.get("dropout", model_config.get("attention_dropout", 0.0))),
            residual=bool(model_config.get("residual", True)),
            layer_norm=bool(model_config.get("layer_norm", True)),
            projection_size=projection_size,
            raw_feature_dim=int(model_config.get("raw_feature_dim", 110)),
        )
    if model_config.get("type") == "paper_bilstm_attention" or "bilstm" in model_config:
        bilstm_config = dict(model_config.get("bilstm", {}))
        attention_config = dict(model_config.get("attention", {}))
        projection_config = dict(model_config.get("projection", {}))
        classifier_config = dict(model_config.get("classifier", {}))
        input_size = int(bilstm_config.get("input_size", 50))
        if input_size != 50:
            raise ValueError("The paper reproduction expects flattened BODY_25 x/y input_size=50")
        configured_classes = int(classifier_config.get("num_classes", num_classes))
        if configured_classes != num_classes:
            raise ValueError(
                f"Configured classifier has {configured_classes} classes, manifest has {num_classes}"
            )
        projection_size = int(projection_config.get("out_features", 64))
        projection_input = int(projection_config.get("in_features", 256))
        hidden_size = int(bilstm_config.get("hidden_size", 128))
        bidirectional = bool(bilstm_config.get("bidirectional", True))
        expected_output = hidden_size * (2 if bidirectional else 1)
        if projection_input != expected_output:
            raise ValueError(
                f"Projection expects {projection_input}, but BiLSTM output is {expected_output}"
            )
        attention_embed_dim = int(attention_config.get("embed_dim", expected_output))
        if attention_embed_dim != expected_output:
            raise ValueError(
                f"Attention embed_dim is {attention_embed_dim}, but BiLSTM output is {expected_output}"
            )
        classifier_input = int(classifier_config.get("in_features", projection_size))
        if classifier_input != projection_size:
            raise ValueError(
                f"Classifier expects {classifier_input}, but projection output is {projection_size}"
            )
        activation = str(projection_config.get("activation", "relu")).lower()
        if activation != "relu":
            raise ValueError("The paper reproduction supports only projection activation=relu")
        pooling_config = model_config.get("pooling", {"type": "temporal_mean"})
        pooling_type = pooling_config.get("type") if isinstance(pooling_config, dict) else pooling_config
        if pooling_type != "temporal_mean":
            raise ValueError("The paper reproduction supports only temporal_mean pooling")
        return PaperBiLSTMAttentionClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=int(bilstm_config.get("num_layers", 2)),
            num_classes=num_classes,
            attention_heads=int(attention_config.get("num_heads", 4)),
            dropout=float(model_config.get("dropout", 0.3)),
            bidirectional=bidirectional,
            lstm_internal_dropout=float(model_config.get("lstm_internal_dropout", 0.0)),
            attention_dropout=float(attention_config.get("dropout", model_config.get("attention_dropout", 0.0))),
            residual=bool(model_config.get("residual", True)),
            layer_norm=bool(model_config.get("layer_norm", True)),
            projection_size=projection_size,
        )
    input_size = int(model_config.get("input_size", 50))
    if input_size != 50:
        raise ValueError("This baseline expects flattened BODY_25 x/y input_size=50")
    return BiLSTMAttentionClassifier(
        input_size=input_size,
        hidden_size=int(model_config.get("hidden_size", 128)),
        num_layers=int(model_config.get("num_layers", 2)),
        num_classes=num_classes,
        attention_heads=int(model_config.get("attention_heads", 4)),
        dropout=float(model_config.get("dropout", 0.3)),
        bidirectional=bool(model_config.get("bidirectional", True)),
        pooling=str(model_config.get("pooling", "mean")),
    )


def class_weights(records: list[dict[str, Any]], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = Counter(int(row["activity_id"]) for row in records)
    total = len(records)
    weights = [total / max(num_classes * counts.get(index, 0), 1) for index in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    amp: bool = False,
    gradient_clip_norm: float | None = None,
    collect_predictions: bool = False,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    batch_loss_sum = 0.0
    num_batches = 0
    prediction_logits = []
    prediction_metadata = []
    y_true: list[int] = []
    y_pred: list[int] = []
    num_classes = len(class_names)
    for batch in loader:
        inputs = batch["skeleton"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        padding_mask = batch["padding_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(inputs, lengths=lengths, padding_mask=padding_mask)
            loss = criterion(logits, labels)
        assert_finite(logits, loss)
        if training:
            if scaler is not None and amp:
                scaler.scale(loss).backward()
                if gradient_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
        total_loss += float(loss.detach().cpu()) * labels.shape[0]
        batch_loss_sum += float(loss.detach().cpu())
        num_batches += 1
        if collect_predictions:
            prediction_logits.extend(logits.detach().cpu().tolist())
            prediction_metadata.extend(batch["metadata"])
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(logits.detach().argmax(dim=1).cpu().tolist())
    overall, _, _ = classification_metrics(y_true, y_pred, num_classes, class_names)
    count = len(y_true)
    return {
        "loss": total_loss / max(count, 1),
        "epoch_loss_mean": total_loss / max(count, 1),
        "epoch_batch_loss_sum": batch_loss_sum,
        "num_batches_per_epoch": num_batches,
        "logits": prediction_logits,
        "metadata": prediction_metadata,
        "accuracy": overall["accuracy"],
        "macro_precision": overall["macro_precision"],
        "macro_recall": overall["macro_recall"],
        "macro_f1": overall["macro_f1"],
        "y_true": y_true,
        "y_pred": y_pred,
    }

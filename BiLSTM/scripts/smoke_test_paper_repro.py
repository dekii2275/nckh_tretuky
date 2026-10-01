#!/usr/bin/env python3
"""Small paper-reproduction data/model forward and backward check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import read_manifest
from src.engine import build_model, class_names_from_records, make_loader, run_epoch
from src.seed import seed_everything
from src.utils import cfg_path, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/paper_repro.yaml")
    parser.add_argument("--train-samples", type=int, default=8)
    parser.add_argument("--val-samples", type=int, default=4)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)), deterministic=True)
    artifacts = cfg_path(config, "outputs", "artifacts_dir", default="artifacts")
    split_dir = cfg_path(config, "outputs", "split_dir", default="artifacts/splits")
    manifest_records = read_manifest(artifacts / "manifest.csv")
    train_records = read_manifest(split_dir / "train.csv")
    val_records = read_manifest(split_dir / "val.csv")
    if not manifest_records or not train_records or not val_records:
        raise RuntimeError("Paper manifest/splits are missing; run prepare_manifest.py first")
    if str(config.get("sequence", {}).get("mode")) != "fixed_60":
        raise AssertionError("Paper smoke test requires sequence.mode=fixed_60")

    class_names = class_names_from_records(manifest_records)
    model = build_model(config, len(class_names))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_loader = make_loader(train_records, config, train=True, max_samples=args.train_samples)
    val_loader = make_loader(val_records, config, train=False, max_samples=args.val_samples)
    batch = next(iter(train_loader))
    skeleton = batch["skeleton"]
    if skeleton.ndim != 4 or tuple(skeleton.shape[1:]) != (60, 25, 2):
        raise AssertionError(f"Expected [B,60,25,2], got {tuple(skeleton.shape)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(
        skeleton.to(device),
        lengths=batch["lengths"].to(device),
        padding_mask=batch["padding_mask"].to(device),
    )
    if logits.shape != (skeleton.shape[0], len(class_names)):
        raise AssertionError(f"Expected [B,10] logits, got {tuple(logits.shape)}")
    loss = criterion(logits, batch["labels"].to(device))
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite smoke-test loss")
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        val_stats = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            class_names,
        )
    for key in ("loss", "accuracy", "macro_f1"):
        if not torch.isfinite(torch.tensor(float(val_stats[key]))):
            raise FloatingPointError(f"Non-finite validation metric: {key}")
    result = {
        "status": "pass",
        "device": str(device),
        "manifest_clips": len(manifest_records),
        "train_samples_used": len(train_loader.dataset),
        "val_samples_used": len(val_loader.dataset),
        "batch_shape": list(skeleton.shape),
        "logits_shape": list(logits.shape),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_loss": float(loss.detach().cpu()),
        "val_loss": val_stats["loss"],
        "val_accuracy": val_stats["accuracy"],
        "val_macro_f1": val_stats["macro_f1"],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

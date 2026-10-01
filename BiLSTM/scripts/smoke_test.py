#!/usr/bin/env python3
"""Fast end-to-end parser -> loader -> model -> backward smoke test."""

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
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/debug.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)), deterministic=True)
    artifacts = cfg_path(config, "outputs", "artifacts_dir", default="artifacts")
    manifest_records = read_manifest(artifacts / "manifest.csv")
    train_records = read_manifest(artifacts / "splits/train.csv")
    val_records = read_manifest(artifacts / "splits/val.csv")
    test_records = read_manifest(artifacts / "splits/test.csv")
    if not manifest_records or not train_records or not val_records or not test_records:
        raise RuntimeError("Manifest/split files are missing or empty; run prepare_manifest.py first")
    class_names = class_names_from_records(manifest_records)
    model = build_model(config, len(class_names))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_config = config.get("training", {})
    train_loader = make_loader(
        train_records,
        config,
        train=True,
        max_samples=int(train_config.get("max_train_samples", 8)),
    )
    val_loader = make_loader(
        val_records,
        config,
        train=False,
        max_samples=int(train_config.get("max_val_samples", 4)),
    )
    batch = next(iter(train_loader))
    skeleton = batch["skeleton"]
    if skeleton.ndim != 4 or tuple(skeleton.shape[-2:]) != (25, 2):
        raise AssertionError(f"Expected [B,T,25,2], got {tuple(skeleton.shape)}")
    with torch.no_grad():
        logits = model(
            skeleton.to(device),
            lengths=batch["lengths"].to(device),
            padding_mask=batch["padding_mask"].to(device),
        )
    if logits.shape != (skeleton.shape[0], len(class_names)):
        raise AssertionError(f"Expected classifier output [B,C], got {tuple(logits.shape)}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_stats = run_epoch(model, train_loader, criterion, device, class_names, optimizer=optimizer)
    with torch.no_grad():
        val_stats = run_epoch(model, val_loader, criterion, device, class_names)
    for stats in (train_stats, val_stats):
        for key in ("loss", "accuracy", "macro_f1"):
            if not torch.isfinite(torch.tensor(float(stats[key]))):
                raise FloatingPointError(f"Non-finite smoke-test metric: {key}")
    result = {
        "status": "pass",
        "device": str(device),
        "manifest_clips": len(manifest_records),
        "train_samples_used": len(train_loader.dataset),
        "val_samples_used": len(val_loader.dataset),
        "batch_shape": list(skeleton.shape),
        "logits_shape": list(logits.shape),
        "train_loss": train_stats["loss"],
        "val_loss": val_stats["loss"],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


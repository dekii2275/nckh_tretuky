"""Train the reproducible MMASD BiLSTM + attention baseline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from .data.dataset import read_manifest
from .engine import build_model, class_names_from_records, class_weights, make_loader, run_epoch
from .seed import seed_everything
from .reproducibility import capture_provenance
from .data.splits import validate_no_leakage
from .utils import cfg_path, ensure_dir, environment_info, json_dump, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint (.pt) to resume training from")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs to train")
    args = parser.parse_args()
    config = load_config(args.config)
    is_window = (
        config.get("sequence", {}).get("mode") == "sliding_window"
        and str(config.get("experiment", "")).startswith("WINDOW_")
    )
    from .sequence_study import is_task3_config, validate_task3_config, clip_metrics_from_stats
    is_task3 = is_task3_config(config)
    if is_task3:
        validate_task3_config(config)
        if not args.resume:
            for key, filename in (("checkpoints_dir", "best.pt"), ("runs_dir", "history.json")):
                if (cfg_path(config, "outputs", key, default="") / filename).exists():
                    raise FileExistsError(f"Refusing to overwrite Task-3 artifact: {key}/{filename}")
    if is_window:
        from .window_study import frozen_inputs, parameter_breakdown
        frozen_inputs(config)
        audit_dir = Path(config["_project_root"]) / "artifacts/window_audit"
        smoke = json.loads((audit_dir / "smoke_tests.json").read_text())
        if len(smoke["experiments"]) != 3 or any(r["status"] != "pass" for r in smoke["experiments"]):
            raise RuntimeError("All three smoke tests must pass before training")
        if not (audit_dir / "window_summary.json").is_file():
            raise RuntimeError("Window count audit is required before training")
        print(json.dumps(environment_info()), flush=True)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA expected but unavailable; no automatic CPU training or package changes")
        if not args.resume:
            for key, filename in (("checkpoints_dir", "best.pt"), ("runs_dir", "history.json"), ("runs_dir", "history_partial.json")):
                if (cfg_path(config, "outputs", key, default="") / filename).exists():
                    raise FileExistsError(f"Refusing to overwrite existing window run: {key}/{filename}")
    seed = int(config.get("seed", 42))
    seed_everything(seed, deterministic=True)
    artifacts = cfg_path(config, "outputs", "artifacts_dir", default="artifacts")
    checkpoint_dir = ensure_dir(cfg_path(config, "outputs", "checkpoints_dir", default="checkpoints"))
    split_dir = cfg_path(config, "outputs", "split_dir", default="artifacts/splits")
    train_path = split_dir / "train.csv"
    val_path = split_dir / "val.csv"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Missing split manifests. Run scripts/prepare_manifest.py first.")
    train_records = read_manifest(train_path)
    val_records = read_manifest(val_path)
    splits = {"train": train_records, "val": val_records}
    if (split_dir / "test.csv").exists():
        splits["test"] = read_manifest(split_dir / "test.csv")
    validate_no_leakage(splits, str(config.get("split", {}).get("strategy", "session_holdout")))
    provenance = capture_provenance(config)
    class_names = class_names_from_records(train_records + val_records)
    model = build_model(config, len(class_names))
    if is_window:
        parameter_breakdown(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    training_config = config.get("training", {})
    weighting = str(training_config.get("class_weighting", "none"))
    weights = class_weights(train_records, len(class_names), device) if weighting == "balanced" else None
    if weighting not in {"none", "balanced"}:
        raise ValueError("training.class_weighting must be none or balanced")
    label_smoothing = float(training_config.get("label_smoothing", 0.0))
    loss_config = training_config.get("loss", {})
    if isinstance(loss_config, str):
        loss_type = loss_config.lower()
        loss_gamma = 2.0
    elif isinstance(loss_config, dict):
        loss_type = str(loss_config.get("type", "CrossEntropyLoss")).lower()
        loss_gamma = float(loss_config.get("gamma", 2.0))
    else:
        loss_type = "crossentropyloss"
        loss_gamma = 2.0

    if loss_type in {"focalloss", "focal"}:
        from .losses import FocalLoss
        criterion = FocalLoss(gamma=loss_gamma, weight=weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    optimizer_config = training_config.get("optimizer", "adam")
    if isinstance(optimizer_config, dict):
        optimizer_name = str(optimizer_config.get("type", "adam")).lower()
        learning_rate = float(optimizer_config.get("learning_rate", 1e-3))
        weight_decay = float(optimizer_config.get("weight_decay", 1e-4))
    else:
        optimizer_name = str(optimizer_config).lower()
        learning_rate = float(training_config.get("learning_rate", 1e-3))
        weight_decay = float(training_config.get("weight_decay", 1e-4))
    if optimizer_name != "adam":
        raise ValueError("Only Adam is implemented for this baseline")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    epochs = int(training_config.get("epochs", 100))
    if args.epochs is not None:
        epochs = int(args.epochs)
    scheduler_type = str(training_config.get("scheduler", "none")).lower()
    scheduler = None
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    amp = bool(training_config.get("amp", True)) and device.type == "cuda"
    if hasattr(torch, "amp"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    train_loader = make_loader(train_records, config, train=True, max_samples=training_config.get("max_train_samples"))
    val_loader = make_loader(val_records, config, train=False, max_samples=training_config.get("max_val_samples"))
    info = environment_info()
    info.update(
        {
            "seed": seed,
            "dataset_size": len(train_records) + len(val_records),
            "train_size": len(train_records),
            "val_size": len(val_records),
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "number_of_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "device": str(device),
            "class_names": class_names,
        }
    )
    print(json.dumps(info, indent=2, ensure_ascii=False))
    early = config.get("early_stopping", {})
    patience = int(early.get("patience", 15))
    monitor = str(early.get("monitor", early.get("metric", "val_macro_f1")))
    if monitor not in {"val_accuracy", "val_macro_f1", "val_loss"}:
        raise ValueError("early_stopping.monitor must be val_accuracy, val_macro_f1, or val_loss")
    mode = str(early.get("mode", "max")).lower()
    if mode not in {"max", "min"}:
        raise ValueError("early_stopping.mode must be max or min")
    best_metric = float("-inf") if mode == "max" else float("inf")
    bad_epochs = 0
    history: list[dict[str, float | int]] = []
    start_epoch = 1

    if args.resume:
        if not args.resume.is_file():
            raise FileNotFoundError(f"Checkpoint for resume not found: {args.resume}")
        resumed = torch.load(args.resume, map_location=device, weights_only=False)
        if "model_state" in resumed:
            model.load_state_dict(resumed["model_state"])
        if "optimizer_state" in resumed and resumed["optimizer_state"] is not None:
            try:
                optimizer.load_state_dict(resumed["optimizer_state"])
            except Exception as exc:
                print(f"Warning: could not restore optimizer state: {exc}")
        if "scaler_state" in resumed and resumed["scaler_state"] is not None and hasattr(scaler, "load_state_dict"):
            try:
                scaler.load_state_dict(resumed["scaler_state"])
            except Exception as exc:
                print(f"Warning: could not restore scaler state: {exc}")
        if "scheduler_state" in resumed and resumed["scheduler_state"] is not None and scheduler is not None:
            try:
                scheduler.load_state_dict(resumed["scheduler_state"])
            except Exception as exc:
                print(f"Warning: could not restore scheduler state: {exc}")
        start_epoch = int(resumed.get("epoch", 0)) + 1
        bad_epochs = int(resumed.get("bad_epochs", 0))
        if "monitor_value" in resumed:
            best_metric = float(resumed["monitor_value"])
        elif "val_macro_f1" in resumed and monitor == "val_macro_f1":
            best_metric = float(resumed["val_macro_f1"])

        if "history" in resumed and isinstance(resumed["history"], list):
            history = [h for h in resumed["history"] if int(h.get("epoch", 0)) < start_epoch]
        else:
            runs_history = cfg_path(config, "outputs", "runs_dir", default="runs") / "history.json"
            if runs_history.is_file():
                try:
                    loaded_runs = json.loads(runs_history.read_text(encoding="utf-8"))
                    if "history" in loaded_runs and isinstance(loaded_runs["history"], list):
                        history = [h for h in loaded_runs["history"] if int(h.get("epoch", 0)) < start_epoch]
                except Exception:
                    pass
        print(
            f"Resumed from {args.resume}: starting at epoch {start_epoch}/{epochs}, "
            f"best {monitor}={best_metric:.6f}, bad_epochs={bad_epochs}"
        )
        if start_epoch > epochs:
            print(f"Training already reached epoch {start_epoch - 1} >= target epochs {epochs}. Nothing to train. (Pass --epochs to train further)")
            return

    gradient_clip = training_config.get("gradient_clip_norm")
    if "gradient_clipping" in training_config:
        gradient_clip = training_config.get("gradient_clipping")
    gradient_clip = float(gradient_clip) if gradient_clip is not None else None
    training_started = time.perf_counter()
    from .data.sequence import SequenceBuilder
    builder = SequenceBuilder(dict(config.get("sequence", {})))
    is_multi_window = builder.is_multi_window
    for epoch in range(start_epoch, epochs + 1):
        epoch_started = time.perf_counter()
        train_stats = run_epoch(
            model, train_loader, criterion, device, class_names, optimizer, scaler, amp,
            gradient_clip,
        )
        with torch.no_grad():
            val_stats = run_epoch(model, val_loader, criterion, device, class_names, amp=amp,
                                  collect_predictions=is_task3 or is_multi_window)
        val_clip = None
        if is_task3 or is_multi_window:
            val_clip, _, _, _ = clip_metrics_from_stats(val_stats, class_names)
        if scheduler is not None:
            scheduler.step()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "epoch_loss_mean": train_stats["epoch_loss_mean"],
            "epoch_batch_loss_sum": train_stats["epoch_batch_loss_sum"],
            "num_batches_per_epoch": train_stats["num_batches_per_epoch"],
            "val_epoch_loss_mean": val_stats["epoch_loss_mean"],
            "val_epoch_batch_loss_sum": val_stats["epoch_batch_loss_sum"],
            "val_num_batches_per_epoch": val_stats["num_batches_per_epoch"],
            "train_accuracy": train_stats["accuracy"],
            "train_macro_f1": train_stats["macro_f1"],
            "val_loss": val_stats["loss"],
            "val_accuracy": val_clip["accuracy"] if val_clip is not None else val_stats["accuracy"],
            "val_macro_f1": val_clip["macro_f1"] if val_clip is not None else val_stats["macro_f1"],
            "val_sequence_accuracy": val_stats["accuracy"],
            "val_sequence_macro_f1": val_stats["macro_f1"],
            "learning_rate": learning_rate,
            "epoch_runtime_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        if is_window or is_task3:
            json_dump({"history": history, "status": "training"},
                      cfg_path(config, "outputs", "runs_dir", default="runs") / "history_partial.json")
        print(json.dumps(row, ensure_ascii=False))
        eval_acc = val_clip["accuracy"] if val_clip is not None else val_stats["accuracy"]
        eval_f1 = val_clip["macro_f1"] if val_clip is not None else val_stats["macro_f1"]
        current = float(
            eval_acc
            if monitor == "val_accuracy"
            else val_stats["loss"]
            if monitor == "val_loss"
            else eval_f1
        )
        improved = current > best_metric if mode == "max" else current < best_metric
        checkpoint_payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "config": config,
            "class_names": class_names,
            "epoch": epoch,
            "bad_epochs": bad_epochs,
            "val_macro_f1": float(val_stats["macro_f1"]),
            "monitor": monitor,
            "monitor_value": best_metric,
            "provenance": provenance,
            "history": history,
        }
        if improved:
            best_metric = current
            bad_epochs = 0
            checkpoint_payload["bad_epochs"] = 0
            checkpoint_payload["monitor_value"] = best_metric
            torch.save(checkpoint_payload, checkpoint_dir / "best.pt")
        else:
            bad_epochs += 1
            checkpoint_payload["bad_epochs"] = bad_epochs

        torch.save(checkpoint_payload, checkpoint_dir / "last.pt")

        if bool(early.get("enabled", True)) and bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch}; best {monitor}={best_metric:.6f}")
            break
    selected = min(history, key=lambda row: row[monitor]) if mode == "min" else max(history, key=lambda row: row[monitor])
    json_dump({
        "environment": info,
        "history": history,
        "monitor": monitor,
        "best_metric": best_metric,
        "best_epoch": selected["epoch"],
        "early_stop_epoch": history[-1]["epoch"] if bad_epochs >= patience else None,
        "last_epoch": history[-1]["epoch"],
        "best_val_macro_f1": selected["val_macro_f1"],
        "selected_checkpoint_metrics": selected,
        "runtime_seconds": time.perf_counter() - training_started,
        "provenance": provenance,
    }, cfg_path(config, "outputs", "runs_dir", default="runs") / "history.json")
    print(f"Best checkpoint: {checkpoint_dir / 'best.pt'}")
    print(f"Last checkpoint: {checkpoint_dir / 'last.pt'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit and smoke all protocols, then optionally run all three on CUDA."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.collate import skeleton_collate
from src.engine import build_model, make_dataset
from src.seed import seed_everything
from src.utils import assert_finite, environment_info, json_dump, load_config
from src.window_study import EXPERIMENTS, audit_windows, frozen_inputs, parameter_breakdown, verify_preserved


def smoke_all(device):
    reports = []
    for stride, experiment in EXPERIMENTS.items():
        config = load_config(ROOT / f"configs/window_s{stride}.yaml")
        splits = frozen_inputs(config)
        seed_everything(config["seed"], deterministic=True)
        model = build_model(config, 10).to(device)
        breakdown = parameter_breakdown(model)
        samples = []
        # One real clip of every class in BOTH splits, plus an overlapping/last window.
        for split, records in splits.items():
            selected = [next(r for r in records if int(r["activity_id"]) == label and int(r["num_frames"]) >= 60)
                        for label in range(10)]
            dataset = make_dataset(selected, config)
            indices = [next(i for i, w in enumerate(dataset.windows) if int(w["activity_id"]) == label) for label in range(10)]
            indices += [len(dataset) - 1]
            for index in indices:
                item = dataset[index]
                meta = dataset.windows[index]
                full = dataset.clips[dataset._clip_indices[meta["clip_id"]]]["skeleton"]
                assert torch.equal(item["skeleton"], full[meta["start_frame"]:meta["end_frame"] + 1])
                assert meta["end_frame"] - meta["start_frame"] == 59
                assert meta["start_frame"] % stride == 0
                assert item["window_id"] == f"{meta['clip_id']}__start{meta['start_frame']:04d}_end{meta['end_frame']:04d}"
                samples.append(item)
        batch = skeleton_collate(samples)
        inputs = batch["skeleton"].to(device)
        assert inputs.shape == (22, 60, 25, 2)
        assert inputs.flatten(2).shape == (22, 60, 50)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
        before = model.classifier.weight.detach().clone()
        logits = model(inputs, lengths=batch["lengths"].to(device), padding_mask=batch["padding_mask"].to(device))
        assert logits.shape == (22, 10)
        loss = torch.nn.CrossEntropyLoss()(logits, batch["labels"].to(device))
        assert_finite(inputs, logits, loss)
        loss.backward()
        for parameter in model.parameters():
            assert parameter.grad is not None
            assert_finite(parameter.grad)
        optimizer.step()
        for parameter in model.parameters():
            assert_finite(parameter)
        assert not torch.equal(before, model.classifier.weight)
        report = {"experiment": experiment, "status": "pass", "device": str(device),
                  "input_shape": list(inputs.shape), "flattened_shape": list(inputs.flatten(2).shape),
                  "logits_shape": list(logits.shape), "loss": float(loss.detach().cpu()),
                  "parameter_breakdown": breakdown, "metadata_and_consecutive_frames": "pass",
                  "backward_and_optimizer_step": "pass", "finite_gradients_parameters": "pass", "leakage": "pass"}
        reports.append(report)
        print(json.dumps(report), flush=True)
    json_dump({"experiments": reports}, ROOT / "artifacts/window_audit/smoke_tests.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Run all three full experiments after audits and smoke tests")
    args = parser.parse_args()
    # CPU thread count is a runtime resource limit, not a model/training change.
    torch.set_num_threads(2)
    info = environment_info()
    print(json.dumps(info, indent=2), flush=True)
    json_dump(info, ROOT / "artifacts/window_audit/cuda_preflight.json")
    summaries = audit_windows()
    print(json.dumps(summaries, indent=2), flush=True)
    smoke_all(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    subprocess.run([sys.executable, str(ROOT / "scripts/audit_confusion_granularity.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/report_window_study.py")], check=True)
    if args.train:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA expected but unavailable. Audits/smokes saved; no packages changed.")
        for stride in EXPERIMENTS:
            frozen_inputs(load_config(ROOT / f"configs/window_s{stride}.yaml"))
            config = f"configs/window_s{stride}.yaml"
            run_dir = ROOT / f"runs/window_s{stride}"
            run_dir.mkdir(parents=True, exist_ok=True)
            # Completed experiments can be reused only after input/checkpoint verification in evaluate.
            if not (run_dir / "history.json").exists():
                with (run_dir / "train.log").open("x") as log:
                    subprocess.run([sys.executable, "-u", "-m", "src.train", "--config", config], cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run([sys.executable, "-u", "-m", "src.evaluate", "--config", config,
                            "--checkpoint", f"checkpoints/window_s{stride}/best.pt"], cwd=ROOT, check=True)
            subprocess.run([sys.executable, str(ROOT / "scripts/audit_confusion_granularity.py")], check=True)
            subprocess.run([sys.executable, str(ROOT / "scripts/report_window_study.py")], check=True)
    verify_preserved()


if __name__ == "__main__":
    main()

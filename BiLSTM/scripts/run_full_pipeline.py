#!/usr/bin/env python3
"""Run the reproducible pipeline end-to-end for one experiment config.

This wrapper intentionally uses the current Python interpreter, so launching it
after ``conda activate mmasd-bilstm`` keeps every stage in that environment.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import cfg_path, load_config


def run_stage(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/paper_repro.yaml")
    parser.add_argument("--no-train", action="store_true", help="Run all preflight stages but skip training/evaluation")
    parser.add_argument("--skip-inspection", action="store_true")
    parser.add_argument("--skip-reconciliation", action="store_true")
    parser.add_argument("--skip-tracking-audit", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--reuse-splits", action="store_true", help="Use existing manifests/splits without regenerating them")
    args = parser.parse_args()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = config_path if config_path.exists() else PROJECT_ROOT / config_path
    args.config = config_path.resolve()
    config = load_config(args.config)

    # The wrapper may run E0 only when explicitly passed its original config;
    # paper/ablation configs must never silently target E0 output directories.
    if args.config.resolve().name != "baseline.yaml":
        e0_paths = {
            cfg_path(config, "outputs", "artifacts_dir", default="artifacts"),
            cfg_path(config, "outputs", "checkpoints_dir", default="checkpoints"),
            cfg_path(config, "outputs", "runs_dir", default="runs"),
        }
        if e0_paths & {
            PROJECT_ROOT / "artifacts",
            PROJECT_ROOT / "checkpoints",
            PROJECT_ROOT / "runs",
        }:
            raise RuntimeError("Refusing to run a non-E0 config against E0 output directories")

    if not args.skip_inspection:
        run_stage(
            [
                sys.executable,
                "scripts/inspect_mmasd.py",
                "--artifacts-dir",
                str(cfg_path(config, "outputs", "artifacts_dir", default="artifacts")),
            ]
        )
    if args.reuse_splits:
        from src.reproducibility import capture_provenance
        capture_provenance(config)  # Fail if required CSV inputs are missing.
    else:
        run_stage([sys.executable, "scripts/prepare_manifest.py", "--config", str(args.config)])
    if not args.skip_reconciliation:
        run_stage([sys.executable, "scripts/reconcile_clip_counts.py"])
    if not args.skip_tracking_audit:
        run_stage([sys.executable, "scripts/audit_tracking.py", "--config", str(args.config)])
    if not args.skip_smoke:
        smoke_script = (
            "scripts/smoke_test_paper_repro.py"
            if str(config.get("sequence", {}).get("mode")) == "fixed_60"
            else "scripts/smoke_test.py"
        )
        run_stage([sys.executable, smoke_script, "--config", str(args.config)])
    if args.no_train:
        print("\nPreflight complete; training was skipped.")
        return

    run_stage([sys.executable, "-m", "src.train", "--config", str(args.config)])
    checkpoint = cfg_path(config, "outputs", "checkpoints_dir", default="checkpoints") / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Training completed without checkpoint: {checkpoint}")
    run_stage(
        [
            sys.executable,
            "-m",
            "src.evaluate",
            "--config",
            str(args.config),
            "--checkpoint",
            str(checkpoint),
            "--split",
            str(config.get("evaluation", {}).get("split", "test")),
        ]
    )
    print("\nFull pipeline completed successfully.")


if __name__ == "__main__":
    main()

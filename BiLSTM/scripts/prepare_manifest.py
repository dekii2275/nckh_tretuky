#!/usr/bin/env python3
"""Create a clip-level manifest and deterministic grouped splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.discover import discover_clips
from src.data.splits import (
    MANIFEST_COLUMNS,
    split_records,
    split_summary,
    split_train_val_records,
    write_csv,
)
from src.utils import cfg_path, ensure_dir, json_dump, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/baseline.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    dataset_root = cfg_path(config, "data", "root", default="data_MMASD")
    skeleton_relative = Path(str(config.get("data", {}).get("skeleton_dir", "2D_skeleton/2D_openpose/output")))
    skeleton_root = skeleton_relative if skeleton_relative.is_absolute() else dataset_root / skeleton_relative
    excluded = list(config.get("classes", {}).get("exclude", []))
    records, diagnostics = discover_clips(skeleton_root, excluded)
    usable = [record.to_dict() for record in records if record.num_frames > 0]
    project_root = Path(config.get("_project_root", PROJECT_ROOT)).resolve()
    for row in usable:
        path = Path(str(row["path"])).resolve()
        try:
            row["path"] = str(path.relative_to(project_root))
        except ValueError:
            row["path"] = str(path)
    if not usable:
        raise RuntimeError("No non-empty clips found; refusing to create a manifest")

    artifacts_dir = cfg_path(config, "outputs", "artifacts_dir", default="artifacts")
    split_dir = cfg_path(config, "outputs", "split_dir", default="artifacts/splits")
    splits_dir = ensure_dir(split_dir)
    write_csv(usable, artifacts_dir / "manifest.csv", columns=list(usable[0].keys()))
    split_config = config.get("split", {})
    if "train_fraction" in split_config and "test_fraction" not in split_config:
        splits = split_train_val_records(
            usable,
            strategy=str(split_config.get("strategy", "session_holdout")),
            seed=int(config.get("seed", 42)),
            val_fraction=float(split_config.get("val_fraction", 0.2)),
        )
    else:
        splits = split_records(
            usable,
            strategy=str(split_config.get("strategy", "session_holdout")),
            seed=int(config.get("seed", 42)),
            val_fraction=float(split_config.get("val_fraction", 0.1)),
            test_fraction=float(split_config.get("test_fraction", 0.2)),
        )
    for name, rows in splits.items():
        write_csv(rows, splits_dir / f"{name}.csv", columns=list(usable[0].keys()))
    metadata = {
        "dataset_root": str(dataset_root),
        "skeleton_root": str(skeleton_root.resolve()),
        "excluded_classes": excluded,
        "all_discovered_classes": diagnostics["all_classes"],
        "included_classes": diagnostics["included_classes"],
        "class_to_id": diagnostics["class_to_id"],
        "empty_clip_count_excluded_from_manifest": diagnostics["empty_clip_count"],
        "manifest_clip_count": len(usable),
        "split": {
            "strategy": split_config.get("strategy", "session_holdout"),
            "seed": int(config.get("seed", 42)),
            "val_fraction": float(split_config.get("val_fraction", 0.1)),
            "train_fraction": float(split_config["train_fraction"])
            if "train_fraction" in split_config
            else None,
            "test_fraction": float(split_config["test_fraction"])
            if "test_fraction" in split_config
            else None,
            "split_dir": str(split_dir),
        },
        "split_summary": split_summary(splits),
    }
    json_dump(metadata, artifacts_dir / "manifest_meta.json")
    json_dump(
        {
            "strategy": metadata["split"]["strategy"],
            "seed": metadata["split"]["seed"],
            "train_fraction": metadata["split"]["train_fraction"],
            "val_fraction": metadata["split"]["val_fraction"],
            "test_fraction": metadata["split"]["test_fraction"],
            "split_summary": metadata["split_summary"],
            "leakage_validation": "passed by split_records/split_train_val_records",
        },
        splits_dir / "split_meta.json",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Saved {artifacts_dir / 'manifest.csv'}")
    saved_splits = ", ".join(str(splits_dir / f"{name}.csv") for name in splits)
    print(f"Saved {saved_splits}")


if __name__ == "__main__":
    main()

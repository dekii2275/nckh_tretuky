"""Audit existing experiment evidence and freeze the MMASD observation pipeline.

Run from repo root: python3 BiLSTM/scripts/audit_pipeline_lock.py
Verify without rewriting: python3 BiLSTM/scripts/audit_pipeline_lock.py --verify
Only the new audit directory is written; source data and old runs are read-only.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import read_manifest  # noqa: E402
from src.data.openpose_parser import parse_clip  # noqa: E402
from src.data.preprocessing import _fill_missing, _valid_mask, preprocess_clip  # noqa: E402
from src.data.tracking import select_person_track  # noqa: E402
from src.engine import make_dataset  # noqa: E402
from src.metrics import classification_metrics  # noqa: E402
from src.reproducibility import inference_settings, sha256_file  # noqa: E402
from src.utils import cfg_path, load_config  # noqa: E402

EXPERIMENTS = {
    "paper_repro": ("paper_repro", "paper_repro"),
    "improved_v1": ("improved_v1", "improved_v1"),
    "improved_v2_center60": ("improved_v2_center60", "improved_v2_center60"),
    "improved_v3_stride30": ("improved_v3_stride30", "improved_v3_stride30"),
    "improved_v4_kinematics": ("improved_v4_kinematics", "improved_v4_kinematics"),
    "improved_v7_focal_loss": ("improved_v7_focal_loss", "improved_v7_focal_loss"),
    "tracker_v1_controlled": ("tracker_training/v1", "paper_repro_tracker_v1"),
    "tracker_v2_controlled": ("tracker_training/v2", "paper_repro_tracker_v2"),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def observe_raw(record: dict[str, Any], config: dict[str, Any]) -> tuple:
    """Return source frames, selected raw x/y/c and pre-imputation validity."""
    check(
        config["audit"]["observation_stage"] == "selected_raw_before_preprocessing",
        "Unsupported audit stage",
    )
    check(config["audit"]["parser_strict"] is True, "Audit requires strict parsing")
    check(config["sequence"]["mode"] == "full_clip", "Raw audit requires full_clip")
    check(
        not config.get("augmentation", {}).get("enabled", False), "Raw audit forbids augmentation"
    )
    # Use configured roots rather than absolute paths recorded on another machine.
    path = (
        cfg_path(config, "data", "root", default="../data_MMASD")
        / config["data"]["skeleton_dir"]
        / record["activity"]
        / record["clip_id"]
    )
    check(path.is_dir(), f"Missing local clip: {path}")
    frames, paths = parse_clip(path, strict=True)
    check(bool(paths), f"Empty clip: {record['clip_id']}")
    check(len(paths) == int(record["num_frames"]), f"Manifest/frame mismatch: {record['clip_id']}")
    data = {**config["data"], "tracking": config["tracking"]}
    threshold = float(
        data.get("confidence_threshold", data["preprocessing"]["confidence_threshold"])
    )
    selection = select_person_track(frames, threshold, data, clip_path=str(path))
    raw = np.full((len(paths), 25, 3), np.nan, dtype=np.float32)
    for index, person in enumerate(selection.selected_people):
        if person is not None:
            raw[index] = person
    return frames, paths, selection, raw, _valid_mask(raw, threshold)


def remember_config(path: Path, remember: Callable[[Path], None], stack: tuple = ()) -> None:
    """Hash every inherited YAML dependency without changing its configuration."""
    path = path.resolve()
    check(path not in stack, f"Circular config dependency: {path}")
    remember(path)
    base = yaml.safe_load(path.read_text(encoding="utf-8")).get("base_config")
    if base:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        remember_config(base_path, remember, (*stack, path))


def environment_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "pyyaml": yaml.__version__,
    }


def verify_snapshot(output: Path, config: dict[str, Any], repo: Path = REPO) -> int:
    """Reject input/output drift without rewriting an existing snapshot."""
    repo = repo.resolve()
    lock = read_json(output / "pipeline_lock.json")
    check(
        lock["schema_version"] in {"skeleton_pipeline_lock_v0_1", "skeleton_pipeline_lock_v0_2"},
        "Unsupported snapshot schema",
    )
    changed = []
    for name, digest in lock["files_sha256"].items():
        path = (repo / name).resolve()
        check(path.is_relative_to(repo), f"Snapshot path outside repository: {name}")
        if not path.is_file() or sha256_file(path) != digest:
            changed.append(name)
    check(not changed, "Frozen files changed: " + ", ".join(changed))
    inventory = sorted(str(p.relative_to(repo)) for p in (repo / "BiLSTM/src").rglob("*.py"))
    check(inventory == lock["source_inventory"], "Source inventory changed")
    current = {k: v for k, v in config.items() if not k.startswith("_")}
    check(current == lock["resolved_config"], "Resolved configuration changed")
    for clip in lock["smoke_frame_inventory"]:
        clip_path = (repo / clip["path"]).resolve()
        check(clip_path.is_relative_to(repo), "Clip path outside repository")
        _, paths = parse_clip(clip_path, strict=True)
        check(
            [str(p.relative_to(repo)) for p in paths] == clip["frames"],
            "Smoke frame inventory changed",
        )
    for name, digest in lock.get("artifacts_sha256", {}).items():
        path = (output / name).resolve()
        check(path.is_relative_to(output.resolve()), "Artifact path outside snapshot")
        check(path.is_file() and sha256_file(path) == digest, f"Snapshot artifact changed: {name}")
    check(environment_versions() == lock["environment"], "Runtime versions changed")
    return len(lock["files_sha256"])


def write_snapshot(output: Path, reports: dict[str, Any], lock: dict[str, Any]) -> None:
    """Prepare all JSON before publishing; never replace an existing version."""
    check(not output.exists(), f"Output exists: {output}. Use --verify or a new version.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pipeline_audit_", dir=output.parent) as temporary:
        staging = Path(temporary)
        for name, report in reports.items():
            write_json(staging / name, report)
        lock = {**lock, "artifacts_sha256": {name: sha256_file(staging / name) for name in reports}}
        write_json(staging / "pipeline_lock.json", lock)
        output.mkdir()
        for path in staging.iterdir():
            path.rename(output / path.name)


def reconcile_experiments(
    validation_ids: set[str],
    validation_labels: dict[str, int],
    canonical_hashes: dict[str, str],
    remember: Callable[[Path], None],
) -> list[dict[str, Any]]:
    experiments = []
    for name, (artifact_name, config_name) in EXPERIMENTS.items():
        result_path = ROOT / "artifacts" / artifact_name / "validation_results.json"
        result = read_json(result_path)
        current_config_path = ROOT / "configs" / f"{config_name}.yaml"
        current_config = load_config(current_config_path)
        remember_config(current_config_path, remember)
        remember(result_path)
        checkpoint_path = (
            cfg_path(current_config, "outputs", "checkpoints_dir", default="checkpoints")
            / "best.pt"
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        remember(checkpoint_path)
        saved = checkpoint["config"]
        history_path = (
            cfg_path(current_config, "outputs", "runs_dir", default="runs") / "history.json"
        )
        history = read_json(history_path)
        remember(history_path)
        epoch = result["checkpoint_epoch"]
        epoch_rows = [row for row in history["history"] if row["epoch"] == epoch]
        row = epoch_rows[0] if len(epoch_rows) == 1 else {}
        accuracy = result.get("clip_accuracy", result.get("accuracy"))
        f1 = result.get("clip_macro_f1", result.get("macro_f1"))
        prediction_path = result_path.parent / "clip_predictions.csv"
        if not prediction_path.exists():
            prediction_path = result_path.parent / "predictions.csv"
        predictions = read_manifest(prediction_path)
        remember(prediction_path)
        check(
            len(predictions) == len(validation_ids)
            and {r["clip_id"] for r in predictions} == validation_ids,
            f"Prediction coverage: {name}",
        )
        check(
            all(int(r["true_id"]) == validation_labels[r["clip_id"]] for r in predictions),
            f"Prediction labels: {name}",
        )
        metrics, _, _ = classification_metrics(
            [int(r["true_id"]) for r in predictions],
            [int(r["predicted_id"]) for r in predictions],
            10,
            result["class_names"],
        )
        checks = {
            "checkpoint_hash_matches_result": sha256_file(checkpoint_path)
            == result["checkpoint_sha256"],
            "checkpoint_epoch_matches_result": checkpoint["epoch"] == epoch,
            "current_inference_config_matches_checkpoint": inference_settings(current_config)
            == inference_settings(saved),
            "evaluation_config_matches_checkpoint": inference_settings(
                result["provenance"]["resolved_config"]
            )
            == inference_settings(saved),
            "checkpoint_manifest_hashes_match_current": checkpoint["provenance"]["manifest_sha256"]
            == canonical_hashes,
            "evaluation_manifest_hashes_match_current": result["provenance"]["manifest_sha256"]
            == canonical_hashes,
            "prediction_accuracy_matches_result": abs(metrics["accuracy"] - accuracy) < 1e-12,
            "prediction_macro_f1_matches_result": abs(metrics["macro_f1"] - f1) < 1e-12,
            "history_accuracy_matches_result": abs(
                row.get("val_clip_accuracy", row.get("val_accuracy", -1)) - accuracy
            )
            < 1e-12,
            "history_macro_f1_matches_result": abs(
                row.get("val_clip_macro_f1", row.get("val_macro_f1", -1)) - f1
            )
            < 1e-12,
        }
        source = checkpoint["provenance"]["source_sha256"]
        changed_source = [
            key
            for key, value in source.items()
            if not (ROOT / "src" / key).exists() or sha256_file(ROOT / "src" / key) != value
        ]
        experiments.append(
            {
                "experiment": name,
                "accuracy": accuracy,
                "macro_f1": f1,
                "correct": sum(r["true_id"] == r["predicted_id"] for r in predictions),
                "clips": len(predictions),
                "epoch": epoch,
                "tracker": result["provenance"]["resolved_config"]
                .get("tracking", {})
                .get("mode", "legacy"),
                "sequence": result["provenance"]["resolved_config"].get("sequence", {}),
                "checks": checks,
                "status": "consistent_saved_run" if all(checks.values()) else "evidence_mismatch",
                "current_checkpoint_epoch": checkpoint["epoch"],
                "current_checkpoint_experiment": saved.get("experiment"),
                "current_checkpoint_model": saved.get("model", {}).get("type"),
                "current_history_best_epoch": history.get("best_epoch"),
                "checkpoint_sha256": result["checkpoint_sha256"],
                "current_checkpoint_sha256": sha256_file(checkpoint_path),
                "checkpoint": relative(checkpoint_path),
                "result": relative(result_path),
                "history": relative(history_path),
                "predictions": relative(prediction_path),
                "source_files_changed_since_training": changed_source,
                "raw_training_data_historically_hashed": False,
            }
        )

    return experiments


def reconcile_stacking(
    validation_ids: set[str], validation_labels: dict[str, int], remember: Callable[[Path], None]
) -> dict[str, Any]:
    # Stacking has saved predictions but no recoverable training procedure here.
    stack_dir = ROOT / "artifacts/stacking_meta_learner_v3_v4_v5_v6"
    stack_result = read_json(stack_dir / "validation_results.json")
    stack_predictions = read_manifest(stack_dir / "clip_predictions.csv")
    for filename in ("validation_results.json", "clip_predictions.csv"):
        remember(stack_dir / filename)
    check(
        len(stack_predictions) == len(validation_ids)
        and {r["clip_id"] for r in stack_predictions} == validation_ids,
        "Stacking coverage",
    )
    check(
        all(int(r["true_id"]) == validation_labels[r["clip_id"]] for r in stack_predictions),
        "Stacking labels",
    )
    stack_metrics, _, _ = classification_metrics(
        [int(r["true_id"]) for r in stack_predictions],
        [int(r["predicted_id"]) for r in stack_predictions],
        10,
        stack_result["class_names"],
    )
    check(
        abs(stack_metrics["accuracy"] - stack_result["clip_accuracy"]) < 1e-12
        and abs(stack_metrics["macro_f1"] - stack_result["clip_macro_f1"]) < 1e-12,
        "Stacking metrics",
    )
    stacking = {
        "metrics": stack_metrics,
        "predictions_match_result": True,
        "method_provenance": "saved predictions only; OOF procedure and meta-model not verified",
        "independent_test": False,
    }

    return stacking


def smoke_pipeline(splits: dict, config: dict[str, Any], remember: Callable[[Path], None]) -> tuple:
    # Technical smoke subset, not a pilot: first train clip of three activities.
    activities = ("arm_swing_as", "drumming_dr", "squat_sq")
    chosen = [
        next(r for r in splits["train"] if r["activity"] == activity) for activity in activities
    ]
    smoke, frame_inventory = [], []
    for record in chosen:
        frames, paths, selection, raw, valid = observe_raw(record, config)
        for path in paths:
            remember(path)
        second = observe_raw(record, config)
        check(selection.selected_indices == second[2].selected_indices, "Non-deterministic tracker")
        check(np.array_equal(raw, second[3], equal_nan=True), "Non-deterministic selected skeleton")
        for frame_index, person_index in enumerate(selection.selected_indices):
            if person_index >= 0:
                check(
                    np.array_equal(
                        raw[frame_index], frames[frame_index][person_index], equal_nan=True
                    ),
                    "Selected x/y/confidence changed",
                )
        pre = config["data"]["preprocessing"]
        processed = preprocess_clip(
            selection.selected_people,
            confidence_threshold=pre["confidence_threshold"],
            missing_joint_strategy=pre["missing_joint_strategy"],
            smoothing=pre["smoothing"],
            min_shoulder_width=pre["shoulder_scale"]["min_width_px"],
            max_interpolation_gap=pre["interpolation"]["max_gap"],
        )
        loaded = make_dataset(
            [{**record, "path": str(paths[0].parent)}], config, return_debug=True
        )[0]
        check(
            loaded["tracking"]["selected_person_per_frame"] == selection.selected_indices,
            "Loader tracker differs",
        )
        check(
            np.array_equal(loaded["skeleton"].numpy(), processed.coordinates),
            "Loader preprocessing differs",
        )
        check(
            np.array_equal(loaded["valid_mask"].numpy(), valid), "Pre-imputation validity differs"
        )
        smoke.append(
            {
                "clip_id": record["clip_id"],
                "activity": record["activity"],
                "frames": len(paths),
                "raw_shape": list(raw.shape),
                "selected_indices": selection.selected_indices,
                "target_missing_frames": selection.selected_indices.count(-1),
                "raw_valid_joints": int(valid.sum()),
                "preprocessing_metadata": processed.metadata,
                "deterministic": True,
                "raw_xy_confidence_preserved": True,
                "loader_tracking_and_processed_coordinates_equal": True,
                "pre_imputation_mask_equal": True,
            }
        )
        frame_inventory.append(
            {"path": relative(paths[0].parent), "frames": [relative(p) for p in paths]}
        )

    return smoke, frame_inventory


def verify_legacy_edge_fill() -> None:
    # Verify the legacy unlimited-gap and edge-fill behavior on a known sequence.
    fixture = np.zeros((7, 25, 3), dtype=np.float32)
    fixture[:, :, :2] = [100, 100]
    fixture[:, :, 2] = 0.9
    fixture[:, 4] = 0
    fixture[2, 4] = [10, 20, 0.9]
    fixture[4, 4] = [30, 40, 0.9]
    filled, mask = _fill_missing(fixture[:, :, :2], _valid_mask(fixture, 0.1), "interpolate", None)
    check(
        np.array_equal(
            filled[:, 4], [[10, 20], [10, 20], [10, 20], [20, 30], [30, 40], [30, 40], [30, 40]]
        ),
        "Edge-fill behavior changed",
    )
    check(mask[:, 4].all(), "Legacy fill mask changed")


def run_audit(config_path: Path, config: dict[str, Any], output: Path) -> None:
    check(not output.exists(), f"Output exists: {output}. Use --verify or a new version.")
    protected: dict[str, str] = {}

    def remember(path: Path) -> None:
        name = relative(path)
        digest = sha256_file(path)
        if name in protected:
            check(protected[name] == digest, f"Input changed while reading: {name}")
        protected[name] = digest

    remember(Path(__file__))
    remember_config(config_path, remember)
    remember(REPO / "docs/reports/BAO_CAO_THUC_NGHIEM_BILSTM.md")
    source_files = sorted((ROOT / "src").rglob("*.py"))
    for path in source_files:
        remember(path)
    manifest_path = ROOT / "artifacts/paper_repro/manifest.csv"
    remember(manifest_path)
    split_dir = ROOT / "artifacts/splits/paper_repro"
    splits = {s: read_manifest(split_dir / f"{s}.csv") for s in ("train", "val")}
    for split in splits:
        remember(split_dir / f"{split}.csv")
    canonical_hashes = {
        "manifest.csv": sha256_file(manifest_path),
        **{f"splits/{s}.csv": sha256_file(split_dir / f"{s}.csv") for s in splits},
    }
    validation_ids = {r["clip_id"] for r in splits["val"]}
    check(len(validation_ids) == len(splits["val"]), "Duplicate validation clip IDs")
    validation_labels = {r["clip_id"]: int(r["activity_id"]) for r in splits["val"]}
    split_summary = {s: len(rows) for s, rows in splits.items()}
    for field in ("clip_id", "subject_id", "session_group"):
        split_summary[f"overlap_{field}"] = sorted(
            {r[field] for r in splits["train"]} & {r[field] for r in splits["val"]}
        )
    check(
        not split_summary["overlap_clip_id"] and not split_summary["overlap_session_group"],
        "Split overlap",
    )

    experiments = reconcile_experiments(
        validation_ids, validation_labels, canonical_hashes, remember
    )
    stacking = reconcile_stacking(validation_ids, validation_labels, remember)
    smoke, frame_inventory = smoke_pipeline(splits, config, remember)
    verify_legacy_edge_fill()
    check(
        all(sha256_file(REPO / name) == digest for name, digest in protected.items()),
        "Read-only audit inputs changed",
    )
    reconciliation = {
        "experiments": experiments,
        "stacking": stacking,
        "split": split_summary,
        "historical_claims_status": "Current saved runs reconciled; unsupported old report values remain unverified. No model inference rerun.",
    }
    smoke_report = {
        "clips": smoke,
        "total_frames": sum(r["frames"] for r in smoke),
        "scope": "3 technical train fixtures, not a representative pilot or tracking accuracy estimate",
        "legacy_unlimited_interpolation_and_edge_fill_verified": True,
    }
    lock = {
        "schema_version": "skeleton_pipeline_lock_v0_2",
        "created_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
        "resolved_config": {k: v for k, v in config.items() if not k.startswith("_")},
        "source_inventory": [relative(p) for p in source_files],
        "files_sha256": protected,
        "smoke_frame_inventory": frame_inventory,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
            "pyyaml": yaml.__version__,
        },
        "raw_data_hash_scope": "all source JSON files of the three smoke clips only",
        "raw_training_data_historically_hashed": False,
        "selected_role_verified": False,
    }
    write_snapshot(
        output,
        {"experiment_reconciliation.json": reconciliation, "smoke_results.json": smoke_report},
        lock,
    )
    mismatches = [r["experiment"] for r in experiments if r["status"] == "evidence_mismatch"]
    print(
        f"PASS: pipeline smoke/freeze; 3 clips/{sum(r['frames'] for r in smoke)} frames; {len(protected)} frozen files"
    )
    print(
        f"Experiment evidence: {len(experiments) - len(mismatches)}/{len(experiments)} consistent; mismatches: {mismatches}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/skeleton_audit_v0_2.yaml")
    parser.add_argument(
        "--output-dir", type=Path, help="New snapshot directory inside the repository"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify without rewriting the snapshot"
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        output = (
            args.output_dir
            or cfg_path(config, "outputs", "artifacts_dir", default="artifacts/pipeline_audit_v0_2")
        ).resolve()
        check(output.is_relative_to(REPO), "Audit output must be inside the repository")
        if args.verify:
            count = verify_snapshot(output, config)
            print(f"PASS: {count} frozen files; outputs, config, inventories and runtime verified")
        else:
            run_audit(args.config, config, output)
        print(relative(output))
        return 0
    except (RuntimeError, OSError, ValueError, KeyError) as error:
        print(f"Audit failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Record the local experiment inputs without claiming author-protocol parity.

Manifest hashes cover CSV contents, not raw skeleton JSON contents. Source
hashes record the implementation used; they are not an author code reference.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .utils import cfg_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inference_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Settings that can change model inputs or predictions at evaluation."""
    data = dict(config.get("data", {}))
    # Caching does not change the sequence values.
    data.pop("cache_in_memory", None)
    return {
        "data": data,
        # Person selection changes the model input before preprocessing, so a
        # checkpoint must never be evaluated with a different tracker mode.
        "tracking": config.get("tracking", {}),
        "sequence": config.get("sequence", {}),
        "model": config.get("model", {}),
        "classes": config.get("classes", {}),
    }


def capture_provenance(config: dict[str, Any]) -> dict[str, Any]:
    split_dir = cfg_path(config, "outputs", "split_dir", default="artifacts/splits")
    artifacts = cfg_path(config, "outputs", "artifacts_dir", default="artifacts")
    if not (artifacts / "manifest.csv").is_file():
        # Ablations reuse the canonical manifest rather than copying and risking drift.
        artifacts = Path(config["_project_root"]) / "artifacts/paper_repro"
    paths = {"manifest.csv": artifacts / "manifest.csv"}
    paths.update({f"splits/{name}.csv": split_dir / f"{name}.csv" for name in ("train", "val", "test")})
    for name in ("manifest.csv", "splits/train.csv", "splits/val.csv"):
        if not paths[name].is_file():
            raise FileNotFoundError(f"Missing reproduction input: {paths[name]}")
    source_root = Path(__file__).resolve().parent
    return {
        "schema_version": 1,
        "manifest_sha256": {name: sha256_file(path) for name, path in paths.items() if path.is_file()},
        "source_sha256": {
            str(path.relative_to(source_root)): sha256_file(path)
            for path in sorted(source_root.rglob("*.py"))
        },
        "raw_skeleton_contents_hashed": False,
        "inference_settings": inference_settings(config),
        "resolved_config": {key: value for key, value in config.items() if not key.startswith("_")},
    }


def verify_checkpoint_inputs(
    checkpoint: dict[str, Any], config: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    saved_config = checkpoint.get("config")
    if saved_config is None:
        raise ValueError("Checkpoint has no config; its preprocessing cannot be verified")
    saved_settings = inference_settings(saved_config)
    differences = [key for key, value in saved_settings.items() if value != inference_settings(config)[key]]
    if differences:
        raise ValueError("Evaluation settings differ from checkpoint: " + ", ".join(differences))
    saved = checkpoint.get("provenance")
    if saved is None:
        return {
            "status": "legacy_checkpoint_manifest_unverified",
            "config_matches": True,
            "note": "This checkpoint predates manifest/source hashes. Current files cannot prove the original split or code.",
        }
    if saved["manifest_sha256"] != current["manifest_sha256"]:
        raise ValueError("Manifest/split contents differ from the training checkpoint")
    return {
        "status": "recorded_manifests_verified",
        "config_matches": True,
        "source_matches": saved["source_sha256"] == current["source_sha256"],
        "raw_skeleton_contents_verified": False,
    }

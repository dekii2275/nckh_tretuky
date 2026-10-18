"""Build a traceable candidate catalogue from selected raw MMASD skeletons."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from audit_pipeline_lock import observe_raw, remember_config  # noqa: E402
from romp_review_data import frame_id  # noqa: E402
from src.data.dataset import read_manifest  # noqa: E402
from src.data.discover import frame_paths  # noqa: E402
from src.reproducibility import sha256_file  # noqa: E402
from src.utils import cfg_path, load_config  # noqa: E402

SCHEMA = "mmasd_pilot_catalog_v1"
DEFAULT_CONFIG = ROOT / "configs/skeleton_audit_v0_2.yaml"
DEFAULT_OUTPUT = ROOT / "artifacts/pilot_selector"


def digest_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def source_digest(paths: list[Path]) -> str:
    """Bind content and ordered source filenames, including frame gaps."""
    return digest_json([[path.name, sha256_file(path)] for path in paths])


def pipeline_hashes(config_path: Path) -> dict[str, str]:
    paths = list((ROOT / "src").rglob("*.py"))
    paths += [Path(__file__), Path(__file__).with_name("audit_pipeline_lock.py")]
    hashes = {str(path.relative_to(REPO)): sha256_file(path) for path in sorted(paths)}
    remember_config(
        config_path, lambda path: hashes.update({str(path.relative_to(REPO)): sha256_file(path)})
    )
    return hashes


def verify_pipeline(catalog: dict[str, Any]) -> None:
    changed = [
        name
        for name, digest in catalog["pipeline_sha256"].items()
        if not (REPO / name).is_file() or sha256_file(REPO / name) != digest
    ]
    if changed:
        raise ValueError("Code/config changed; rebuild the catalogue: " + ", ".join(changed))


def max_missing_run(valid: np.ndarray, source_ids: list[int]) -> int:
    """Count observed source slots; never bridge a missing source file."""
    longest = 0
    current = np.zeros(valid.shape[1], dtype=np.int64)
    for index, observed in enumerate(valid):
        if index and source_ids[index] != source_ids[index - 1] + 1:
            current[:] = 0
        current = np.where(observed, 0, current + 1)
        longest = max(longest, int(current.max()))
    return longest


def quality_band(rate: float) -> str:
    # Missingness strata only: these are not ground-truth good/bad pose labels.
    if rate <= 0.05:
        return "low_missing"
    return "medium_missing" if rate <= 0.20 else "high_missing"


def measure_raw(
    raw: np.ndarray,
    valid: np.ndarray,
    source_ids: list[int],
    people_counts: list[int],
    threshold: float,
) -> dict[str, Any]:
    if len(source_ids) != len(set(source_ids)) or source_ids != sorted(source_ids):
        raise ValueError("Source frame IDs are duplicated or unordered")
    confidence = raw[:, :, 2]
    finite_confidence = np.isfinite(confidence)
    confidence_count = int(finite_confidence.sum())
    missing = float((~valid).mean())
    diagonals = []
    for frame, mask in zip(raw, valid):
        if mask.sum() >= 2:
            diagonal = float(np.linalg.norm(np.ptp(frame[mask, :2], axis=0)))
            if diagonal > 1e-6:
                diagonals.append(diagonal)
    scale = float(np.median(diagonals)) if diagonals else None
    large_step_count, max_step = 0, None
    if scale is not None and len(raw) > 1:
        pair_valid = valid[1:] & valid[:-1]
        pair_valid &= (np.diff(source_ids) == 1)[:, None]
        steps = np.linalg.norm(raw[1:, :, :2] - raw[:-1, :, :2], axis=2) / scale
        values = steps[pair_valid]
        if len(values):
            max_step = float(values.max())
            large_step_count = int((values > 0.35).sum())
    return {
        "missing_joint_ratio": missing,
        "missingness_band": quality_band(missing),
        "mean_confidence": float(confidence[finite_confidence].mean())
        if confidence_count
        else None,
        "low_confidence_ratio": float((confidence[finite_confidence] < threshold).mean())
        if confidence_count
        else None,
        "mean_valid_confidence": float(confidence[valid].mean()) if valid.any() else None,
        "max_missing_run": max_missing_run(valid, source_ids),
        "target_missing_frames": int((~np.isfinite(raw).any(axis=(1, 2))).sum()),
        "source_gap_count": int(np.maximum(np.diff(source_ids) - 1, 0).sum()),
        "source_first_frame": source_ids[0],
        "source_last_frame": source_ids[-1],
        "mean_people": float(np.mean(people_counts)),
        "max_people": max(people_counts),
        "large_step_joint_events": large_step_count,
        "max_step_body_diagonals": max_step,
        "reference_body_diagonal": scale,
    }


def read_observation(record: dict[str, Any], config: dict[str, Any]) -> tuple:
    directory = (
        cfg_path(config, "data", "root", default="../data_MMASD")
        / config["data"]["skeleton_dir"]
        / record["activity"]
        / record["clip_id"]
    )
    before_paths = frame_paths(directory)
    before = [(path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in before_paths]
    frames, paths, tracking, raw, valid = observe_raw(record, config)
    after = [
        (path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in frame_paths(directory)
    ]
    if before != after:
        raise ValueError("Source frame files changed while reading the clip")
    ids = [frame_id(path) for path in paths]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate source frame IDs")
    return frames, paths, tracking, raw, valid, ids


def build_catalog(
    config_path: Path, manifest: Path, output: Path, limit: int = 0
) -> dict[str, Any]:
    config = load_config(config_path)
    if config["tracking"]["mode"] != "tracklet_v2":
        raise ValueError("The pilot catalogue requires the frozen Tracklet V2 configuration")
    protected = pipeline_hashes(config_path)
    manifest_hash = sha256_file(manifest)
    records = read_manifest(manifest)
    if len({row["clip_id"] for row in records}) != len(records):
        raise ValueError("Duplicate clip IDs in candidate manifest")
    selected = records[:limit] if limit > 0 else records
    rows = []
    threshold = float(config["data"]["preprocessing"]["confidence_threshold"])
    for index, record in enumerate(selected):
        row = {
            key: record[key]
            for key in (
                "clip_id",
                "activity",
                "activity_id",
                "subject_id",
                "session_id",
                "num_frames",
            )
        }
        row["num_frames"] = int(row["num_frames"])
        try:
            if not record["subject_id"].strip() or record["subject_id"] == "unknown":
                raise ValueError("Missing subject identifier")
            frames, paths, _, raw, valid, ids = read_observation(record, config)
            if not valid.any():
                raise ValueError(
                    "No valid selected skeleton; excluded from the pilot candidate pool"
                )
            row.update(measure_raw(raw, valid, ids, [len(frame) for frame in frames], threshold))
            row.update(status="ready", source_sha256=source_digest(paths))
        except (ValueError, OSError, KeyError, TypeError) as error:
            row.update(status="quarantined", error=str(error))
        rows.append(row)
        if (index + 1) % 10 == 0 or index + 1 == len(selected):
            print(
                f"Catalogue {index + 1}/{len(selected)}; quarantined={sum(r['status'] != 'ready' for r in rows)}",
                flush=True,
            )
    if pipeline_hashes(config_path) != protected or sha256_file(manifest) != manifest_hash:
        raise ValueError(
            "Pipeline/manifest changed during catalogue build; outputs were not published"
        )
    catalog = {
        "schema_version": SCHEMA,
        "created_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
        "config_path": str(config_path.resolve().relative_to(REPO)),
        "config_sha256": sha256_file(config_path),
        "pipeline_sha256": protected,
        "manifest_sha256": manifest_hash,
        "population_size": len(records),
        "measured_size": len(rows),
        "complete_population": len(rows) == len(records),
        "tracker": "tracklet_v2",
        "confidence_threshold": threshold,
        "definitions": {
            "missing_joint_ratio": "Invalid selected joints / (source frames * 25); absent target counts as missing.",
            "low_confidence_ratio": "Finite confidence values below threshold / all finite selected confidence values; includes source zeros.",
            "max_missing_run": "Longest invalid joint run in available consecutive source frames, including edges; not across missing source files.",
            "missingness_bands": "low <= 5%; medium <= 20%; high > 20%; missingness strata, not pose correctness.",
            "large_step_joint_events": "Valid adjacent-source joint displacement > 0.35 median body diagonals; motion indicator, not verified error.",
            "identity": "Anonymous selected track; child identity is unverified.",
            "fps": "Unverified. Metrics use source frames, not seconds.",
        },
        "rows": rows,
    }
    catalog["catalog_sha256"] = digest_json(catalog)
    output.mkdir(parents=True, exist_ok=True)
    archives = output / "catalogues"
    archives.mkdir(exist_ok=True)
    if (output / "catalog.json").exists():
        previous = load_catalog(output / "catalog.json")
        previous_path = archives / (previous["catalog_sha256"] + ".json")
        if not previous_path.exists():
            previous_path.write_bytes((output / "catalog.json").read_bytes())
    temporary = output / "catalog.json.tmp"
    temporary.write_text(json.dumps(catalog, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "catalog.json")
    archive_path = archives / (catalog["catalog_sha256"] + ".json")
    if not archive_path.exists():
        archive_path.write_bytes((output / "catalog.json").read_bytes())
    return catalog


def load_catalog(path: Path) -> dict[str, Any]:
    catalog = json.loads(path.read_text())
    expected = catalog.pop("catalog_sha256")
    if catalog.get("schema_version") != SCHEMA or digest_json(catalog) != expected:
        raise ValueError("Catalogue schema or checksum does not match")
    catalog["catalog_sha256"] = expected
    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "artifacts/paper_repro/manifest.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Technical smoke subset; default measures all candidates",
    )
    args = parser.parse_args()
    result = build_catalog(args.config.resolve(), args.manifest, args.output_dir, args.limit)
    print(f"Ready: {result['measured_size']} clips; {args.output_dir / 'catalog.json'}")


if __name__ == "__main__":
    main()

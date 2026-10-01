"""Shared read-only experiment inputs and diagnostic artifact helpers."""
import json
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reproducibility import sha256_file
from src.utils import json_dump
from src.window_study import write_csv

OUT = ROOT / "artifacts/diagnostics"
REVIEW = ROOT / "artifacts/tracking_review"
PAPER = dict(zip([
    "arm_swing_as", "body_swing_bs", "chest_expansion_ce", "drumming_dr",
    "maracas_forward_shaking_mfs", "maracas_shaking_ms", "sing_and_clap_sac",
    "squat_sq", "tree_pose_tr", "twist_pose_tw"],
    [96.55, 92.88, 98.67, 98.51, 97.17, 81.10, 76.33, 96.01, 94.06, 99.78], strict=True))
NAMES = list(PAPER)
PAIRS = {(1, 2): 5.36, (5, 4): 9.42, (6, 3): 20.39}


def read_manifest(path):
    # Browser CSV exports may include a UTF-8 BOM.
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def preserve_runs():
    paths = []
    for directory in ("artifacts/paper_repro", "artifacts/splits/paper_repro", "runs/paper_repro", "checkpoints/paper_repro",
                      "artifacts/window_s60", "artifacts/window_s30", "artifacts/window_s15",
                      "runs/window_s60", "runs/window_s30", "runs/window_s15",
                      "checkpoints/window_s60", "checkpoints/window_s30", "checkpoints/window_s15"):
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.is_file())
    paths.extend((ROOT / "src").rglob("*.py"))
    paths.extend((ROOT / "configs").glob("*.yaml"))
    hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(paths)}
    target = OUT / "protected_experiments_sha256.json"
    if target.exists():
        if hashes != json.loads(target.read_text()):
            raise RuntimeError("Protected R0/S1/S2/S3, configuration, split or source changed")
    else:
        json_dump(hashes, target)
    return len(hashes)


def split_records():
    from src.data.splits import validate_no_leakage
    splits = {s: read_manifest(ROOT / f"artifacts/splits/paper_repro/{s}.csv") for s in ("train", "val")}
    validate_no_leakage(splits, "session_holdout")
    return splits


def read_labels(path=None):
    path = Path(path) if path else REVIEW / "manual_labels.csv"
    rows = read_manifest(path) if path.exists() else []
    seen = set()
    expected = {r["clip_id"]: r for r in read_manifest(REVIEW / "review_manifest.csv")}
    for row in rows:
        key = row["clip_id"]
        if key in seen or key not in expected:
            raise ValueError(f"Unknown/duplicate reviewed clip: {key}")
        seen.add(key)
        label = row.get("review_label", "").strip().upper()
        if label not in {"", "A", "B", "C", "D"}:
            raise ValueError(f"Invalid human label for {key}: {label}")
        row["review_label"] = label
        for field in ("multiple_people", "partial_occlusion", "identity_switch_visible", "child_leaves_frame", "therapist_crosses_child", "severe_keypoint_failure", "tiny_skeleton", "large_camera_motion"):
            if row.get(field, "") not in {"", "0", "1"}:
                raise ValueError(f"Invalid optional human flag: {key}/{field}")
        for field in ("subject_id", "session_id", "activity", "num_frames", "max_people"):
            if str(row[field]) != str(expected[key][field]):
                raise ValueError(f"Review metadata differs from frozen review manifest: {key}/{field}")
    return rows


def markdown_table(rows, columns):
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        values = ["N/A" if row.get(k) is None else f"{row[k]:.3f}" if isinstance(row[k], float) else str(row[k]) for k in columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines

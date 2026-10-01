#!/usr/bin/env python3
"""Deterministic Dryad -> plain MSG3D preprocessing and split builder.

Round 3 intentionally stops at a real-batch, forward-only smoke test.  This
module never trains, mutates raw XLSX files, repairs the value 100, or creates
official paper folds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import platform
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from audit_dryad_duplicates import digest_rows, scan  # noqa: E402
from dryad_contract import (  # noqa: E402
    CONTRACT_VERSION,
    SENTINEL_VALUE,
    TARGET_T,
    stack_zero_person,
)
from dryad_joint_mapping import MODEL_TO_RAW, RAW_TO_MODEL, RAW_JOINTS  # noqa: E402
from investigate_dryad_anomalies import (  # noqa: E402
    read_sheet,
    skeleton_files,
    source_metadata,
)


DATASET_ROOT_DEFAULT = (
    REPO_ROOT
    / "Data_DRYAD"
    / "doi_10_5061_dryad_s7h44j150__v20200911"
    / "Dataset-2"
    / "Dataset"
)
OUTPUT_ROOT_DEFAULT = REPO_ROOT / "artifacts" / "dryad_msg3d"
CLASS_MAPPING_VERSION = "CLASS_TD0_ASD1_V1"
JOINT_MAPPING_VERSION = "DRYAD_NTU25_V1"
PREPROCESSING_VERSION = "DRYAD_MSG3D_PREPROCESS_V1"
FOLD_COUNT = 10
REPRODUCTION_SEED = 2026091701
LEAKAGE_SAFE_SEED = 2026091702
PROTOCOLS = {
    "random_reproduction_like": REPRODUCTION_SEED,
    "random_leakage_safe": LEAKAGE_SAFE_SEED,
}
MANIFEST_FIELDS = [
    "sample_index",
    "sample_id",
    "class_name",
    "class_id",
    "subject_id",
    "source",
    "is_original",
    "is_augmented",
    "augmentation_id",
    "original_sample_id",
    "source_path",
    "raw_sha256",
    "numeric_hash",
    "raw_frame_count",
    "valid_frame_count",
    "target_frame_count",
    "raw_joint_count",
    "raw_coordinate_count",
    "coord_order",
    "joint_mapping_version",
    "preprocessing_version",
    "m_policy",
    "had_sentinel",
    "sentinel_count",
    "sentinel_joints",
    "sentinel_components",
    "repair_method",
    "duplicate_group_id",
    "duplicate_component_id",
]


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value}")


def subject_key(class_name: str, subject_id: str) -> str:
    return f"{class_name}/{subject_id}"


def subject_sort_key(key: str) -> tuple[int, int, str]:
    class_name, subject_id = key.split("/", 1)
    class_rank = {"ASD": 0, "TD": 1}.get(class_name, 2)
    try:
        numeric_id = int(subject_id)
    except ValueError:
        numeric_id = 10**9
    return class_rank, numeric_id, subject_id


def normalized_subject_sort_key(key: str) -> tuple[int, int, str]:
    class_name, subject_id = key.split("_", 1)
    class_rank = {"ASD": 0, "TD": 1}.get(class_name, 2)
    try:
        numeric_id = int(subject_id)
    except ValueError:
        numeric_id = 10**9
    return class_rank, numeric_id, subject_id


def normalize_component_subject(key: str) -> str:
    class_name, raw_subject_id = key.split("/", 1)
    return f"{class_name}_{int(raw_subject_id):03d}"


def path_sort_key(path: Path) -> tuple[int, int, int, int, str]:
    metadata = source_metadata(path)
    class_rank = {"ASD": 0, "TD": 1}.get(str(metadata["class"]), 2)
    subject_text = str(metadata["subject_id"])
    try:
        subject_number = int(subject_text)
    except ValueError:
        subject_number = 10**9
    source_rank = 0 if metadata["source"] == "original" else 1
    augmentation_id = metadata["augmentation_id"]
    augmentation_number = 0 if augmentation_id is None else int(str(augmentation_id))
    return class_rank, subject_number, source_rank, augmentation_number, path.as_posix()


class DisjointSet:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def canonical_path(path: str | Path) -> str:
    return str(Path(path).resolve())


def relative_source_path(path: Path, dataset_root: Path) -> str:
    return path.resolve().relative_to(dataset_root.resolve()).as_posix()


def read_raw_coordinates(path: Path) -> np.ndarray:
    """Read exactly the first 25*3 coordinate cells as [T,25,3] float64."""

    rows = read_sheet(path)
    if len(rows) < 2:
        raise ValueError(f"{path}: expected a header and at least one frame")
    coordinate_rows: list[list[float]] = []
    for frame_index, row in enumerate(rows[1:]):
        if len(row) < 1 + 25 * 3:
            raise ValueError(
                f"{path}: frame {frame_index} has fewer than 75 coordinate cells"
            )
        values = row[1 : 1 + 25 * 3]
        if not all(isinstance(value, (int, float)) for value in values):
            raise ValueError(f"{path}: nonnumeric coordinate at frame {frame_index}")
        numeric = [float(value) for value in values]
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f"{path}: non-finite coordinate at frame {frame_index}")
        coordinate_rows.append(numeric)
    raw = np.asarray(coordinate_rows, dtype=np.float64).reshape(-1, 25, 3)
    if raw.shape[1:] != (25, 3):
        raise ValueError(f"{path}: unexpected raw shape {raw.shape}")
    if raw.shape[0] > TARGET_T:
        raise ValueError(f"{path}: T_raw={raw.shape[0]} exceeds TARGET_T={TARGET_T}")
    if raw.shape[0] < 1:
        raise ValueError(f"{path}: empty skeleton")
    return raw


def mapped_to_model(raw: np.ndarray) -> np.ndarray:
    """Apply raw-index -> model-index mapping without changing coordinates."""

    mapped = np.empty_like(raw)
    for raw_index, model_index in enumerate(RAW_TO_MODEL):
        mapped[:, model_index, :] = raw[:, raw_index, :]
    return mapped


def preprocess_array(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mapped [T,25,3], padded [60,25,3], and final [3,60,25,2]."""

    mapped = mapped_to_model(raw)
    padded = np.zeros((TARGET_T, 25, 3), dtype=np.float32)
    padded[: raw.shape[0], :, :] = mapped
    final = stack_zero_person(padded.transpose(2, 0, 1), target_t=TARGET_T)
    return mapped, padded, final


def sentinel_details(raw: np.ndarray) -> tuple[int, list[str], list[int]]:
    mask = raw == SENTINEL_VALUE
    joints = sorted(
        {
            RAW_JOINTS[joint_index]
            for joint_index, component_index in zip(*np.where(mask.any(axis=0)))
        }
    )
    components = sorted({int(index) for index in np.where(mask)[2]})
    return int(mask.sum()), joints, components


def build_duplicate_lookup(dataset_root: Path) -> dict[str, object]:
    """Rebuild the deterministic duplicate graph used by Round 2 artifacts."""

    report = scan(dataset_root)
    numeric_groups = report["numeric_duplicate_groups"]
    path_to_group: dict[str, str] = {}
    path_to_numeric_hash: dict[str, str] = {}
    group_subjects: dict[str, set[str]] = {}
    for group_index, group in enumerate(numeric_groups, start=1):
        group_id = f"NG{group_index:03d}"
        subjects: set[str] = set()
        for record in group["files"]:
            path_key = canonical_path(str(record["file"]))
            path_to_group[path_key] = group_id
            path_to_numeric_hash[path_key] = str(record["numeric_digest"])
            subjects.add(subject_key(str(record["class"]), str(record["subject_id"])))
        group_subjects[group_id] = subjects

    files = sorted(skeleton_files(dataset_root), key=path_sort_key)
    all_subjects = sorted(
        {
            subject_key(str(metadata["class"]), str(metadata["subject_id"]))
            for path in files
            for metadata in [source_metadata(path)]
        },
        key=subject_sort_key,
    )
    dsu = DisjointSet(all_subjects)
    for subjects in group_subjects.values():
        ordered = sorted(subjects, key=subject_sort_key)
        for other in ordered[1:]:
            dsu.union(ordered[0], other)

    components: dict[str, list[str]] = defaultdict(list)
    for subject in all_subjects:
        components[dsu.find(subject)].append(subject)
    ordered_components = sorted(
        components.values(),
        key=lambda members: subject_sort_key(sorted(members, key=subject_sort_key)[0]),
    )
    component_by_subject: dict[str, str] = {}
    component_members: dict[str, list[str]] = {}
    for index, members in enumerate(ordered_components, start=1):
        component_id = f"C{index:03d}"
        ordered_members = sorted(members, key=subject_sort_key)
        component_members[component_id] = ordered_members
        for member in ordered_members:
            component_by_subject[member] = component_id

    for path in files:
        path_key = canonical_path(path)
        metadata = source_metadata(path)
        subject = subject_key(str(metadata["class"]), str(metadata["subject_id"]))
        if path_key not in path_to_numeric_hash:
            path_to_numeric_hash[path_key] = digest_rows(read_sheet(path))

    return {
        "path_to_group": path_to_group,
        "path_to_numeric_hash": path_to_numeric_hash,
        "component_by_subject": component_by_subject,
        "component_members": component_members,
    }


def build_manifest(dataset_root: Path) -> list[dict[str, object]]:
    files = sorted(skeleton_files(dataset_root), key=path_sort_key)
    if len(files) != 800:
        raise ValueError(f"expected 800 skeleton files, found {len(files)}")
    duplicate = build_duplicate_lookup(dataset_root)
    rows: list[dict[str, object]] = []
    for sample_index, path in enumerate(files):
        metadata = source_metadata(path)
        class_name = str(metadata["class"])
        if class_name not in {"ASD", "TD"}:
            raise ValueError(f"cannot parse class from {path}")
        raw_subject = str(metadata["subject_id"])
        try:
            subject_number = int(raw_subject)
        except ValueError as exc:
            raise ValueError(f"cannot parse subject from {path}") from exc
        subject_id = f"{class_name}_{subject_number:03d}"
        original_sample_id = f"{subject_id}_original"
        source = str(metadata["source"])
        is_original = source == "original"
        augmentation_id = "" if is_original else f"{int(str(metadata['augmentation_id'])):02d}"
        sample_id = original_sample_id if is_original else f"{subject_id}_aug_{augmentation_id}"
        if not is_original and original_sample_id == sample_id:
            raise AssertionError("augmentation sample ID collided with original")

        raw = read_raw_coordinates(path)
        count, joints, components = sentinel_details(raw)
        path_key = canonical_path(path)
        subject_component = duplicate["component_by_subject"][
            subject_key(class_name, raw_subject)
        ]
        rows.append(
            {
                "sample_index": sample_index,
                "sample_id": sample_id,
                "class_name": class_name,
                "class_id": 1 if class_name == "ASD" else 0,
                "subject_id": subject_id,
                "source": source,
                "is_original": "1" if is_original else "0",
                "is_augmented": "0" if is_original else "1",
                "augmentation_id": augmentation_id,
                "original_sample_id": original_sample_id,
                "source_path": relative_source_path(path, dataset_root),
                "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "numeric_hash": duplicate["path_to_numeric_hash"][path_key],
                "raw_frame_count": raw.shape[0],
                "valid_frame_count": raw.shape[0],
                "target_frame_count": TARGET_T,
                "raw_joint_count": 25,
                "raw_coordinate_count": 3,
                "coord_order": "XYZ",
                "joint_mapping_version": JOINT_MAPPING_VERSION,
                "preprocessing_version": PREPROCESSING_VERSION,
                "m_policy": "ZERO_PAD_SECOND_PERSON",
                "had_sentinel": "1" if count else "0",
                "sentinel_count": count,
                "sentinel_joints": "|".join(joints),
                "sentinel_components": "|".join(str(value) for value in components),
                "repair_method": "none",
                "duplicate_group_id": duplicate["path_to_group"].get(path_key, ""),
                "duplicate_component_id": subject_component,
            }
        )

    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id collision")
    if {str(row["subject_id"]) for row in rows} != {
        *(f"ASD_{index:03d}" for index in range(1, 51)),
        *(f"TD_{index:03d}" for index in range(1, 51)),
    }:
        raise ValueError("unexpected subject set")
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    write_csv(path, MANIFEST_FIELDS, rows)


def roundtrip_metrics(raw: np.ndarray, final: np.ndarray, mapped: np.ndarray) -> dict[str, float]:
    t_raw = raw.shape[0]
    valid_model = final[:, :t_raw, :, 0].transpose(1, 2, 0)
    recovered_raw = np.empty_like(valid_model)
    for raw_index, model_index in enumerate(RAW_TO_MODEL):
        recovered_raw[:, raw_index, :] = valid_model[:, model_index, :]
    raw_float32 = raw.astype(np.float32)
    diff_cast = np.abs(recovered_raw - raw_float32)
    mapped_diff = np.abs(valid_model - mapped.astype(np.float32))
    return {
        "max_abs_error_vs_float32_raw": float(diff_cast.max(initial=0.0)),
        "mean_abs_error_vs_float32_raw": float(diff_cast.mean()),
        "max_abs_error_vs_float64_raw": float(np.abs(recovered_raw.astype(np.float64) - raw).max(initial=0.0)),
        "mean_abs_error_vs_float64_raw": float(np.abs(recovered_raw.astype(np.float64) - raw).mean()),
        "valid_region_max_abs_diff": float(mapped_diff.max(initial=0.0)),
    }


def preprocess_path(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    raw = read_raw_coordinates(path)
    mapped, padded, final = preprocess_array(raw)
    if final.shape != (3, TARGET_T, 25, 2):
        raise AssertionError(f"unexpected final shape: {final.shape}")
    metrics = roundtrip_metrics(raw, final, mapped)
    t_raw = raw.shape[0]
    if t_raw < TARGET_T and np.count_nonzero(padded[t_raw:]) != 0:
        raise AssertionError(f"nonzero temporal padding in {path}")
    if np.count_nonzero(final[:, :, :, 1]) != 0:
        raise AssertionError(f"nonzero person1 in {path}")
    raw_sentinel_count = int(np.count_nonzero(raw == SENTINEL_VALUE))
    final_sentinel_count = int(np.count_nonzero(final[:, :t_raw, :, 0] == SENTINEL_VALUE))
    if raw_sentinel_count != final_sentinel_count:
        raise AssertionError(f"sentinel count changed in {path}")
    metrics.update(
        {
            "raw_frame_count": t_raw,
            "raw_sentinel_count": raw_sentinel_count,
            "person0_nonzero_count": int(np.count_nonzero(final[:, :, :, 0])),
            "person1_nonzero_count": int(np.count_nonzero(final[:, :, :, 1])),
            "padded_region_nonzero_count": int(np.count_nonzero(final[:, t_raw:, :, 0])),
        }
    )
    return raw, mapped, final, metrics


def select_golden_samples(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def choose(name: str, predicate) -> dict[str, object]:
        matches = [row for row in rows if predicate(row)]
        if not matches:
            raise ValueError(f"cannot select golden sample: {name}")
        selected = sorted(matches, key=lambda row: int(row["sample_index"]))[0].copy()
        selected["golden_category"] = name
        return selected

    selected: list[dict[str, object]] = []
    selected.append(
        choose(
            "ASD original without sentinel",
            lambda row: row["class_name"] == "ASD"
            and row["is_original"] == "1"
            and row["had_sentinel"] == "0",
        )
    )
    selected.append(
        choose(
            "ASD original with sentinel",
            lambda row: row["class_name"] == "ASD"
            and row["is_original"] == "1"
            and row["had_sentinel"] == "1",
        )
    )
    selected.append(
        choose(
            "TD original",
            lambda row: row["class_name"] == "TD" and row["is_original"] == "1",
        )
    )
    selected.append(
        choose(
            "ASD augmentation",
            lambda row: row["class_name"] == "ASD" and row["is_augmented"] == "1",
        )
    )
    selected.append(
        choose(
            "TD augmentation",
            lambda row: row["class_name"] == "TD" and row["is_augmented"] == "1",
        )
    )
    selected.append(
        choose("shortest-T sample", lambda row: int(row["raw_frame_count"]) == 21)
    )
    selected.append(choose("longest-T sample", lambda row: int(row["raw_frame_count"]) == 60))
    return selected


def verify_golden_samples(
    rows: list[dict[str, object]], dataset_root: Path, print_results: bool = True
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for row in select_golden_samples(rows):
        path = dataset_root / str(row["source_path"])
        raw, mapped, final, metrics = preprocess_path(path)
        result = {
            "golden_category": row["golden_category"],
            "sample_id": row["sample_id"],
            "source_path": row["source_path"],
            "class_name": row["class_name"],
            "subject_id": row["subject_id"],
            "augmentation_id": row["augmentation_id"],
            "T_raw": int(raw.shape[0]),
            "sentinel_count": int(metrics["raw_sentinel_count"]),
            "raw_shape": list(raw.shape),
            "mapped_shape": list(mapped.shape),
            "padded_shape": [TARGET_T, 25, 3],
            "final_shape": list(final.shape),
            **metrics,
        }
        results.append(result)
        if print_results:
            print(
                "GOLDEN",
                row["golden_category"],
                row["sample_id"],
                f"T={raw.shape[0]}",
                f"sentinel={metrics['raw_sentinel_count']}",
                f"raw={raw.shape}",
                f"mapped={mapped.shape}",
                f"padded={(TARGET_T, 25, 3)}",
                f"final={final.shape}",
                f"person1_nonzero={metrics['person1_nonzero_count']}",
                f"padded_nonzero={metrics['padded_region_nonzero_count']}",
                f"valid_diff={metrics['valid_region_max_abs_diff']:.3g}",
            )
    return results


def process_all(
    rows: list[dict[str, object]], dataset_root: Path
) -> tuple[np.ndarray, dict[str, object]]:
    data = np.empty((len(rows), 3, TARGET_T, 25, 2), dtype=np.float32)
    raw_t_counts: Counter[int] = Counter()
    coordinate_count = 0
    coordinate_sum = np.zeros(3, dtype=np.float64)
    coordinate_sumsq = np.zeros(3, dtype=np.float64)
    coordinate_min = np.full(3, np.inf, dtype=np.float64)
    coordinate_max = np.full(3, -np.inf, dtype=np.float64)
    total_sentinel = 0
    total_padded_frames = 0
    nan_count = 0
    inf_count = 0
    roundtrip_max_cast = 0.0
    roundtrip_mean_sum = 0.0
    roundtrip_element_count = 0
    sentinel_samples = 0
    for index, row in enumerate(rows):
        if int(row["sample_index"]) != index:
            raise AssertionError("manifest sample_index is not contiguous")
        path = dataset_root / str(row["source_path"])
        raw, mapped, final, metrics = preprocess_path(path)
        data[index] = final
        raw_t_counts[raw.shape[0]] += 1
        total_padded_frames += TARGET_T - raw.shape[0]
        total_sentinel += int(metrics["raw_sentinel_count"])
        sentinel_samples += int(metrics["raw_sentinel_count"] > 0)
        nan_count += int(np.isnan(final).sum())
        inf_count += int(np.isinf(final).sum())
        roundtrip_max_cast = max(roundtrip_max_cast, float(metrics["max_abs_error_vs_float32_raw"]))
        roundtrip_mean_sum += float(metrics["mean_abs_error_vs_float32_raw"])
        roundtrip_element_count += int(raw.size)

        flat = raw.astype(np.float64).reshape(-1, 3)
        coordinate_count += flat.shape[0]
        coordinate_sum += flat.sum(axis=0)
        coordinate_sumsq += np.square(flat).sum(axis=0)
        coordinate_min = np.minimum(coordinate_min, flat.min(axis=0))
        coordinate_max = np.maximum(coordinate_max, flat.max(axis=0))

    coordinate_mean = coordinate_sum / coordinate_count
    coordinate_variance = np.maximum(coordinate_sumsq / coordinate_count - coordinate_mean**2, 0.0)
    coordinate_std = np.sqrt(coordinate_variance)
    stats = {
        "tensor_shape": list(data.shape),
        "dtype": str(data.dtype),
        "raw_frame_distribution": {str(key): value for key, value in sorted(raw_t_counts.items())},
        "raw_t_min": min(raw_t_counts),
        "raw_t_max": max(raw_t_counts),
        "total_padded_frames": total_padded_frames,
        "padded_frame_percentage": total_padded_frames / (len(rows) * TARGET_T),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "sentinel_occurrences_valid_data": total_sentinel,
        "sentinel_sample_count": sentinel_samples,
        "coordinate_statistics_including_sentinel": {
            "x": {
                "min": float(coordinate_min[0]),
                "max": float(coordinate_max[0]),
                "mean": float(coordinate_mean[0]),
                "std": float(coordinate_std[0]),
            },
            "y": {
                "min": float(coordinate_min[1]),
                "max": float(coordinate_max[1]),
                "mean": float(coordinate_mean[1]),
                "std": float(coordinate_std[1]),
            },
            "z": {
                "min": float(coordinate_min[2]),
                "max": float(coordinate_max[2]),
                "mean": float(coordinate_mean[2]),
                "std": float(coordinate_std[2]),
            },
        },
        "roundtrip_max_abs_error_vs_float32_raw": roundtrip_max_cast,
        "roundtrip_mean_abs_error_vs_float32_raw": roundtrip_mean_sum / len(rows),
        "roundtrip_element_count": roundtrip_element_count,
    }
    if tuple(data.shape) != (800, 3, 60, 25, 2):
        raise AssertionError(f"unexpected tensor shape: {data.shape}")
    if data.dtype != np.float32:
        raise AssertionError(f"unexpected tensor dtype: {data.dtype}")
    if int(np.count_nonzero(data[:, :, :, :, 1])) != 0:
        raise AssertionError("person 1 is not all zero")
    if int(np.count_nonzero(data == SENTINEL_VALUE)) != 101:
        raise AssertionError("serialized tensor sentinel count is not 101")
    return data, stats


def class_subjects(rows: list[dict[str, object]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows:
        subject_id = str(row["subject_id"])
        class_name = str(row["class_name"])
        if subject_id in result and result[subject_id] != class_name:
            raise AssertionError("subject belongs to multiple classes")
        result[subject_id] = class_name
    return result


def indices_for_subjects(rows: list[dict[str, object]], subjects: set[str]) -> list[int]:
    return [int(row["sample_index"]) for row in rows if str(row["subject_id"]) in subjects]


def indices_for_components(rows: list[dict[str, object]], components: set[str]) -> list[int]:
    return [
        int(row["sample_index"])
        for row in rows
        if str(row["duplicate_component_id"]) in components
    ]


def assign_subject_folds(subjects_by_class: dict[str, list[str]], seed: int) -> list[set[str]]:
    folds = [set() for _ in range(FOLD_COUNT)]
    for class_offset, class_name in enumerate(("ASD", "TD")):
        values = sorted(subjects_by_class[class_name], key=normalized_subject_sort_key)
        random.Random(seed + class_offset).shuffle(values)
        for index, subject in enumerate(values):
            folds[index % FOLD_COUNT].add(subject)
    return folds


def assign_component_folds(
    rows: list[dict[str, object]], component_members: dict[str, list[str]], seed: int
) -> list[set[str]]:
    subjects_to_class = class_subjects(rows)
    component_class: dict[str, str] = {}
    for component_id, members in component_members.items():
        classes = set()
        for subject in members:
            class_name, raw_subject_id = subject.split("/", 1)
            normalized_subject_id = f"{class_name}_{int(raw_subject_id):03d}"
            classes.add(subjects_to_class[normalized_subject_id])
        if len(classes) != 1:
            raise AssertionError("duplicate component crosses class boundary")
        component_class[component_id] = next(iter(classes))

    fold_components = [set() for _ in range(FOLD_COUNT)]
    for class_offset, class_name in enumerate(("ASD", "TD")):
        components = [
            component_id
            for component_id, component_class_name in component_class.items()
            if component_class_name == class_name
        ]
        rng = random.Random(seed + class_offset)
        rng.shuffle(components)
        components.sort(key=lambda component_id: -len(component_members[component_id]))
        subject_counts = [0] * FOLD_COUNT
        for component_id in components:
            minimum = min(subject_counts)
            candidates = [index for index, count in enumerate(subject_counts) if count == minimum]
            fold_index = rng.choice(candidates)
            fold_components[fold_index].add(component_id)
            subject_counts[fold_index] += len(component_members[component_id])

    fold_subjects: list[set[str]] = []
    for component_set in fold_components:
        fold_subjects.append(
            {
                normalize_component_subject(subject)
                for component_id in component_set
                for subject in component_members[component_id]
            }
        )
    return fold_subjects


def fold_stats(
    rows: list[dict[str, object]],
    train_indices: list[int],
    test_indices: list[int],
    train_subject_ids: list[str],
    test_subject_ids: list[str],
    protocol: str,
    fold_id: int,
    seed: int,
) -> dict[str, object]:
    hashes = {
        "train": {str(rows[index]["numeric_hash"]) for index in train_indices if rows[index]["numeric_hash"]},
        "test": {str(rows[index]["numeric_hash"]) for index in test_indices if rows[index]["numeric_hash"]},
    }
    train_components = {str(rows[index]["duplicate_component_id"]) for index in train_indices}
    test_components = {str(rows[index]["duplicate_component_id"]) for index in test_indices}
    hash_intersection = hashes["train"] & hashes["test"]
    component_intersection = train_components & test_components

    def count(field: str, value: str, indices: list[int]) -> int:
        return sum(str(rows[index][field]) == value for index in indices)

    result: dict[str, object] = {
        "fold_id": fold_id,
        "protocol": protocol,
        "seed": seed,
        "train_subject_count": len(train_subject_ids),
        "test_subject_count": len(test_subject_ids),
        "train_subject_ids": sorted(train_subject_ids, key=normalized_subject_sort_key),
        "test_subject_ids": sorted(test_subject_ids, key=normalized_subject_sort_key),
        "train_sample_count": len(train_indices),
        "test_sample_count": len(test_indices),
        "train_class_counts": {
            "ASD": count("class_name", "ASD", train_indices),
            "TD": count("class_name", "TD", train_indices),
        },
        "test_class_counts": {
            "ASD": count("class_name", "ASD", test_indices),
            "TD": count("class_name", "TD", test_indices),
        },
        "train_original_count": count("is_original", "1", train_indices),
        "test_original_count": count("is_original", "1", test_indices),
        "train_augmented_count": count("is_augmented", "1", train_indices),
        "test_augmented_count": count("is_augmented", "1", test_indices),
        "train_component_count": len(train_components),
        "test_component_count": len(test_components),
        "duplicate_leakage_count": len(hash_intersection),
        "train_test_component_intersection": sorted(component_intersection),
        "train_test_numeric_hash_intersection": sorted(hash_intersection),
        "train_subject_intersection": sorted(set(train_subject_ids) & set(test_subject_ids)),
        "train_sample_ids": [str(rows[index]["sample_id"]) for index in train_indices],
        "test_sample_ids": [str(rows[index]["sample_id"]) for index in test_indices],
        "train_indices": train_indices,
        "test_indices": test_indices,
        "train_duplicate_components": sorted(train_components),
        "test_duplicate_components": sorted(test_components),
        "train_numeric_hashes": sorted(hashes["train"]),
        "test_numeric_hashes": sorted(hashes["test"]),
    }
    return result


def generate_folds(
    rows: list[dict[str, object]],
    component_members: dict[str, list[str]],
    output_root: Path,
    selected_protocol: str,
) -> dict[str, list[dict[str, object]]]:
    subjects_by_class: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        subject_id = str(row["subject_id"])
        class_name = str(row["class_name"])
        if subject_id not in subjects_by_class[class_name]:
            subjects_by_class[class_name].append(subject_id)
    protocol_names = list(PROTOCOLS) if selected_protocol == "all" else [selected_protocol]
    generated: dict[str, list[dict[str, object]]] = {}
    for protocol in protocol_names:
        seed = PROTOCOLS[protocol]
        if protocol == "random_reproduction_like":
            fold_subjects = assign_subject_folds(subjects_by_class, seed)
            split_unit = "subject_id"
            algorithm = "class-wise seeded shuffle, round-robin into 10 folds"
        else:
            fold_subjects = assign_component_folds(rows, component_members, seed)
            split_unit = "duplicate_component_id"
            algorithm = "seeded component shuffle, largest-first greedy balance within class"
        protocol_dir = output_root / "folds" / protocol
        protocol_dir.mkdir(parents=True, exist_ok=True)
        summaries: list[dict[str, object]] = []
        for fold_id, test_subject_set in enumerate(fold_subjects):
            test_indices = sorted(indices_for_subjects(rows, test_subject_set))
            train_indices = sorted(set(range(len(rows))) - set(test_indices))
            test_subject_ids = sorted(test_subject_set, key=normalized_subject_sort_key)
            train_subject_ids = sorted(
                set(str(row["subject_id"]) for row in rows) - test_subject_set,
                key=normalized_subject_sort_key,
            )
            summary = fold_stats(
                rows,
                train_indices,
                test_indices,
                train_subject_ids,
                test_subject_ids,
                protocol,
                fold_id,
                seed,
            )
            summary["split_unit"] = split_unit
            summary["balancing_algorithm"] = algorithm
            if summary["train_subject_intersection"]:
                raise AssertionError(f"subject leakage in {protocol} fold {fold_id}")
            if set(summary["train_sample_ids"]) & set(summary["test_sample_ids"]):
                raise AssertionError(f"sample leakage in {protocol} fold {fold_id}")
            if protocol == "random_leakage_safe":
                if summary["train_test_component_intersection"]:
                    raise AssertionError(f"component leakage in fold {fold_id}")
                if summary["train_test_numeric_hash_intersection"]:
                    raise AssertionError(f"numeric-hash leakage in fold {fold_id}")
            write_json(protocol_dir / f"fold_{fold_id:02d}.json", summary)
            summaries.append(summary)
            print(
                "FOLD",
                protocol,
                f"fold={fold_id:02d}",
                f"train_subjects={len(train_subject_ids)}",
                f"test_subjects={len(test_subject_ids)}",
                f"train_ASD={summary['train_class_counts']['ASD']}",
                f"train_TD={summary['train_class_counts']['TD']}",
                f"test_ASD={summary['test_class_counts']['ASD']}",
                f"test_TD={summary['test_class_counts']['TD']}",
                f"train_samples={len(train_indices)}",
                f"test_samples={len(test_indices)}",
                f"duplicate_leakage={summary['duplicate_leakage_count']}",
            )
        summary_rows = []
        for summary in summaries:
            summary_rows.append(
                {
                    "fold": summary["fold_id"],
                    "protocol": protocol,
                    "train_subject_count": summary["train_subject_count"],
                    "test_subject_count": summary["test_subject_count"],
                    "train_ASD": summary["train_class_counts"]["ASD"],
                    "train_TD": summary["train_class_counts"]["TD"],
                    "test_ASD": summary["test_class_counts"]["ASD"],
                    "test_TD": summary["test_class_counts"]["TD"],
                    "train_samples": summary["train_sample_count"],
                    "test_samples": summary["test_sample_count"],
                    "train_original": summary["train_original_count"],
                    "test_original": summary["test_original_count"],
                    "train_augmented": summary["train_augmented_count"],
                    "test_augmented": summary["test_augmented_count"],
                    "train_components": summary["train_component_count"],
                    "test_components": summary["test_component_count"],
                    "duplicate_leakage_count": summary["duplicate_leakage_count"],
                }
            )
        write_csv(
            protocol_dir / "fold_summary.csv",
            list(summary_rows[0]),
            summary_rows,
        )
        generated[protocol] = summaries
    return generated


def write_loader_artifacts(
    rows: list[dict[str, object]],
    data: np.ndarray,
    folds: dict[str, list[dict[str, object]]],
    output_root: Path,
) -> None:
    for protocol, summaries in folds.items():
        loader_dir = output_root / "loader" / protocol
        noaug_dir = output_root / "noaug" / protocol
        loader_dir.mkdir(parents=True, exist_ok=True)
        noaug_dir.mkdir(parents=True, exist_ok=True)
        for summary in summaries:
            fold_id = int(summary["fold_id"])
            train_indices = np.asarray(summary["train_indices"], dtype=np.int64)
            test_indices = np.asarray(summary["test_indices"], dtype=np.int64)
            for split_name, indices in (("train", train_indices), ("test", test_indices)):
                np.save(loader_dir / f"{split_name}_data_fold_{fold_id:02d}.npy", data[indices])
                labels = np.asarray(
                    [int(rows[int(index)]["class_id"]) for index in indices], dtype=np.int64
                )
                sample_paths = [str(rows[int(index)]["source_path"]) for index in indices]
                with (loader_dir / f"{split_name}_labels_fold_{fold_id:02d}.pkl").open("wb") as handle:
                    pickle.dump((labels, sample_paths), handle, protocol=4)

            noaug_indices = np.asarray(
                [
                    int(index)
                    for index in test_indices
                    if rows[int(index)]["is_original"] == "1"
                ],
                dtype=np.int64,
            )
            np.save(noaug_dir / f"test_data_NoAug_fold_{fold_id:02d}.npy", data[noaug_indices])
            noaug_labels = np.asarray(
                [int(rows[int(index)]["class_id"]) for index in noaug_indices], dtype=np.int64
            )
            noaug_paths = [str(rows[int(index)]["source_path"]) for index in noaug_indices]
            with (noaug_dir / f"test_labels_NoAug_fold_{fold_id:02d}.pkl").open("wb") as handle:
                pickle.dump((noaug_labels, noaug_paths), handle, protocol=4)
            write_json(
                noaug_dir / f"fold_{fold_id:02d}.json",
                {
                    "fold_id": fold_id,
                    "protocol": protocol,
                    "test_subject_ids": summary["test_subject_ids"],
                    "test_original_only": True,
                    "noaug_indices": noaug_indices.tolist(),
                    "noaug_sample_ids": [
                        str(rows[int(index)]["sample_id"]) for index in noaug_indices
                    ],
                    "selection_rule": "original samples belonging to test subjects only",
                },
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"git_commit": commit, "git_dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unavailable", "git_dirty": None}


def write_checksums(output_root: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.name == "checksums.json":
            continue
        checksums[path.relative_to(output_root).as_posix()] = sha256_file(path)
    write_json(output_root / "checksums.json", checksums)
    return checksums


class SerializedFoldDataset:
    """Minimal contract-compatible loader for one serialized split."""

    def __init__(self, data_path: Path, labels_path: Path):
        self.data = np.load(data_path, mmap_mode="r")
        with labels_path.open("rb") as handle:
            self.labels, self.sample_paths = pickle.load(handle)
        if len(self.data) != len(self.labels) or len(self.data) != len(self.sample_paths):
            raise ValueError("loader artifact lengths disagree")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            np.array(self.data[index], dtype=np.float32, copy=True),
            int(self.labels[index]),
            self.sample_paths[index],
        )


def run_forward_smoke(output_root: Path, protocol: str) -> dict[str, object]:
    import torch
    from torch.utils.data import DataLoader, Dataset

    autism_root = REPO_ROOT / "autism"
    if str(autism_root) not in sys.path:
        sys.path.insert(0, str(autism_root))
    from plain_msg3d import build_plain_msg3d, load_ntu120_backbone

    class TorchDataset(Dataset):
        def __init__(self, serialized: SerializedFoldDataset):
            self.serialized = serialized

        def __len__(self):
            return len(self.serialized)

        def __getitem__(self, index):
            data, label, source_path = self.serialized[index]
            return torch.from_numpy(data), label, source_path

    loader_root = output_root / "loader" / protocol
    serialized = SerializedFoldDataset(
        loader_root / "train_data_fold_00.npy",
        loader_root / "train_labels_fold_00.pkl",
    )
    if len(serialized) == 0:
        raise AssertionError("fold 00 train loader is empty")
    loader = DataLoader(TorchDataset(serialized), batch_size=4, shuffle=False, num_workers=0)
    batch_data, batch_labels, batch_paths = next(iter(loader))
    if tuple(batch_data.shape[1:]) != (3, 60, 25, 2):
        raise AssertionError(f"bad loader batch shape: {tuple(batch_data.shape)}")
    if not torch.isfinite(batch_data).all():
        raise AssertionError("non-finite value in real loader batch")
    if torch.count_nonzero(batch_data[:, :, :, :, 1]) != 0:
        raise AssertionError("person1 is nonzero in real loader batch")
    if not set(batch_labels.tolist()).issubset({0, 1}):
        raise AssertionError("unexpected label in real loader batch")

    torch.manual_seed(20260917)
    model = build_plain_msg3d(num_class=2, num_person=2)
    checkpoint = REPO_ROOT / "autism" / "msg3d_pretrained_weights" / "ntu120-xsub-joint.pt"
    load_info = load_ntu120_backbone(model, checkpoint)
    model.eval()
    captured: dict[str, torch.Tensor] = {}

    def capture_fc_input(_module, arguments):
        captured["feature"] = arguments[0].detach().cpu()

    hook = model.fc.register_forward_pre_hook(capture_fc_input)
    with torch.no_grad():
        logits = model(batch_data)
    hook.remove()
    if tuple(logits.shape) != (batch_data.shape[0], 2):
        raise AssertionError(f"bad logits shape: {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise AssertionError("non-finite logits")
    feature = captured.get("feature")
    if feature is None or not torch.isfinite(feature).all():
        raise AssertionError("missing/non-finite pre-FC feature")
    result = {
        "protocol": protocol,
        "fold": 0,
        "batch_data_shape": list(batch_data.shape),
        "batch_label_shape": list(batch_labels.shape),
        "batch_labels": batch_labels.tolist(),
        "batch_source_paths": list(batch_paths),
        "person1_absmax": float(batch_data[:, :, :, :, 1].abs().max()),
        "feature_shape": list(feature.shape),
        "logits_shape": list(logits.shape),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "feature_finite": bool(torch.isfinite(feature).all()),
        "checkpoint_load": load_info,
        "training_performed": False,
        "optimizer_step_performed": False,
    }
    write_json(output_root / "real_batch_forward_smoke.json", result)
    print(
        "FORWARD_SMOKE",
        f"protocol={protocol}",
        f"data={tuple(batch_data.shape)}",
        f"labels={batch_labels.tolist()}",
        f"feature={tuple(feature.shape)}",
        f"logits={tuple(logits.shape)}",
        "finite=True",
    )
    return result


def assert_output_policy(output_root: Path, overwrite: bool) -> None:
    generated = [
        output_root / "manifest.csv",
        output_root / "preprocessed" / "all_data.npy",
        output_root / "folds",
        output_root / "loader",
        output_root / "noaug",
        output_root / "checksums.json",
        output_root / "build_metadata.json",
    ]
    existing = [path for path in generated if path.exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing[:5])
        raise FileExistsError(
            f"generated artifacts already exist ({rendered}); use --overwrite true explicitly"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--protocol",
        choices=["all", *PROTOCOLS],
        default="all",
    )
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)
    if args.dry_run and args.verify:
        raise ValueError("--dry-run and --verify cannot be combined")
    if not args.dry_run:
        assert_output_policy(output_root, args.overwrite)

    print(f"DATASET_ROOT={dataset_root}")
    print(f"OUTPUT_ROOT={output_root}")
    print(f"CONTRACT={CONTRACT_VERSION}")
    print(f"CLASS_MAPPING_VERSION={CLASS_MAPPING_VERSION} TD=0 ASD=1")
    print(f"SEEDS={PROTOCOLS}")
    rows = build_manifest(dataset_root)
    golden = verify_golden_samples(rows, dataset_root)
    if args.dry_run:
        print(
            "DRY_RUN_PASS",
            f"manifest_rows={len(rows)}",
            f"ASD={sum(row['class_name'] == 'ASD' for row in rows)}",
            f"TD={sum(row['class_name'] == 'TD' for row in rows)}",
        )
        return

    output_root.mkdir(parents=True, exist_ok=True)
    write_manifest(output_root / "manifest.csv", rows)
    write_json(
        output_root / "preprocessing_config.json",
        {
            "contract_version": CONTRACT_VERSION,
            "class_mapping_version": CLASS_MAPPING_VERSION,
            "class_mapping": {"TD": 0, "ASD": 1},
            "joint_mapping_version": JOINT_MAPPING_VERSION,
            "raw_to_model": list(RAW_TO_MODEL),
            "model_to_raw": list(MODEL_TO_RAW),
            "coord_order": "XYZ",
            "target_t": TARGET_T,
            "m": 2,
            "person_policy": "ZERO_PAD_SECOND_PERSON",
            "sentinel_policy": "PRESERVE_RAW_WITH_FLAG",
            "normalization": "none",
            "protocol_seeds": PROTOCOLS,
            "source_dataset_root": str(dataset_root),
            "preprocessing_version": PREPROCESSING_VERSION,
        },
    )
    write_json(output_root / "golden_samples.json", golden)

    data, tensor_stats = process_all(rows, dataset_root)
    preprocessed_dir = output_root / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    np.save(preprocessed_dir / "all_data.npy", data)
    labels = np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64)
    np.save(preprocessed_dir / "all_labels.npy", labels)
    write_json(
        preprocessed_dir / "all_sample_ids.json",
        [str(row["sample_id"]) for row in rows],
    )
    reloaded = np.load(preprocessed_dir / "all_data.npy")
    if not np.array_equal(data, reloaded):
        raise AssertionError("all_data.npy did not round-trip exactly")
    tensor_stats["serialization_exact_equality"] = True
    write_json(output_root / "tensor_stats.json", tensor_stats)

    duplicate = build_duplicate_lookup(dataset_root)
    folds = generate_folds(
        rows,
        duplicate["component_members"],
        output_root,
        args.protocol,
    )
    write_loader_artifacts(rows, data, folds, output_root)
    smoke_protocol = "random_reproduction_like" if "random_reproduction_like" in folds else next(iter(folds))
    smoke = run_forward_smoke(output_root, smoke_protocol)

    metadata = {
        **git_metadata(),
        "contract_version": CONTRACT_VERSION,
        "class_mapping_version": CLASS_MAPPING_VERSION,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pytorch": __import__("torch").__version__,
        "seeds": PROTOCOLS,
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "manifest_rows": len(rows),
        "tensor_stats": tensor_stats,
        "fold_protocols": sorted(folds),
        "real_batch_forward_smoke": smoke,
        "training_performed": False,
        "optimizer_step_performed": False,
        "official_folds_created": False,
    }
    write_json(output_root / "build_metadata.json", metadata)
    checksums = write_checksums(output_root)
    print(
        "ROUND3_BUILD_PASS",
        f"manifest_rows={len(rows)}",
        f"tensor_shape={tuple(data.shape)}",
        f"tensor_dtype={data.dtype}",
        f"checksum_files={len(checksums)}",
        f"protocols={sorted(folds)}",
    )
    if args.verify:
        verify_outputs(output_root, rows, data, folds, smoke_protocol)
        print("ROUND3_VERIFY_PASS")


def verify_outputs(
    output_root: Path,
    rows: list[dict[str, object]],
    data: np.ndarray,
    folds: dict[str, list[dict[str, object]]],
    smoke_protocol: str,
) -> None:
    manifest_path = output_root / "manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if len(manifest_rows) != len(rows):
        raise AssertionError("manifest reload row count mismatch")
    reloaded = np.load(output_root / "preprocessed" / "all_data.npy")
    if not np.array_equal(data, reloaded):
        raise AssertionError("verification reload mismatch")
    for protocol, summaries in folds.items():
        if len(summaries) != FOLD_COUNT:
            raise AssertionError(f"{protocol}: expected 10 folds")
        for summary in summaries:
            if summary["train_subject_intersection"]:
                raise AssertionError(f"{protocol}: subject leakage")
            if protocol == "random_leakage_safe":
                if summary["train_test_component_intersection"]:
                    raise AssertionError(f"{protocol}: component leakage")
                if summary["train_test_numeric_hash_intersection"]:
                    raise AssertionError(f"{protocol}: numeric hash leakage")
    if smoke_protocol not in folds:
        raise AssertionError("smoke protocol was not generated")


if __name__ == "__main__":
    main()

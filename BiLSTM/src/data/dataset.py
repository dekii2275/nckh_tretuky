"""PyTorch dataset for one full OpenPose clip per sample."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .openpose_parser import parse_clip
from .preprocessing import preprocess_clip
from .tracking import select_person_track
from .sequence import SequenceBuilder, temporal_resample
from .augmentation import SkeletonAugmentation


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sample_indices(length: int, target: int, mode: str) -> np.ndarray:
    if target <= 0:
        raise ValueError("target_frames/max_frames must be positive")
    if length <= target:
        return np.arange(length, dtype=np.int64)
    if mode == "uniform_sample":
        return np.linspace(0, length - 1, target, dtype=np.int64)
    if mode == "fixed_window":
        start = (length - target) // 2
        return np.arange(start, start + target, dtype=np.int64)
    return np.linspace(0, length - 1, target, dtype=np.int64)


def uniform_temporal_resample(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """Deterministically convert a ``[T, J, 2]`` sequence to target length.

    Longer clips use uniformly spaced source frames. Shorter clips use linear
    interpolation along time; this is an implementation choice because the
    paper specifies 60 frames but does not describe the conversion method.
    """
    return temporal_resample(sequence, target_length)


class SkeletonClipDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        records: Iterable[dict[str, Any]],
        data_config: dict[str, Any] | None = None,
        sequence_config: dict[str, Any] | None = None,
        return_debug: bool = False,
        train: bool = False,
    ) -> None:
        self.records = [dict(record) for record in records if int(record.get("num_frames", 0)) > 0]
        self.data_config = data_config or {}
        self.sequence_config = sequence_config or {}
        self.sequence_builder = SequenceBuilder(self.sequence_config)
        self.return_debug = return_debug
        self.train = train
        augmentation_config = dict(self.data_config.get("augmentation", {}))
        self.augmentation = SkeletonAugmentation(augmentation_config)
        self._rng = np.random.default_rng(int(self.data_config.get("seed", 42)))
        self.cache_in_memory = bool(self.data_config.get("cache_in_memory", True))
        self._sequence_cache: dict[int, np.ndarray] = {}
        if not self.records:
            raise ValueError("Dataset has no non-empty clip records")
        self.samples = self.sequence_builder.expand(self.records)
        if len(self.samples) != len(self.records):
            raise ValueError("SkeletonClipDataset requires exactly one sequence per source clip")

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_path(self, path: str) -> Path:
        value = Path(path)
        if value.is_absolute():
            if value.exists():
                return value
            # Fallback for portability if dataset path recorded on another machine
            parts = value.parts
            if "data_MMASD" in parts:
                candidate = Path(*parts[parts.index("data_MMASD"):])
                root = Path(self.data_config.get("project_root", Path.cwd()))
                if (root / candidate).exists():
                    return (root / candidate).resolve()
                if (root.parent / candidate).exists():
                    return (root.parent / candidate).resolve()
            return value
        root = Path(self.data_config.get("project_root", Path.cwd()))
        return (root / value).resolve()

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        sequence_metadata = self.samples[index]
        clip_path = self._resolve_path(str(record["path"]))

        tracking = None
        preprocessing = None
        sequence = (
            self._sequence_cache.get(index)
            if self.cache_in_memory and not self.return_debug
            else None
        )

        if sequence is None:
            frames, _ = parse_clip(clip_path, strict=True)
            preprocessing_config = dict(self.data_config.get("preprocessing", {}))
            shoulder_config = dict(
                self.data_config.get("shoulder_scale", preprocessing_config.get("shoulder_scale", {}))
            )
            interpolation_config = dict(
                self.data_config.get("interpolation", preprocessing_config.get("interpolation", {}))
            )
            confidence_threshold = float(
                self.data_config.get(
                    "confidence_threshold", preprocessing_config.get("confidence_threshold", 0.1)
                )
            )
            missing_joint_strategy = str(
                self.data_config.get(
                    "missing_joint_strategy", preprocessing_config.get("missing_joint_strategy", "interpolate")
                )
            )
            tracking = select_person_track(frames, confidence_threshold, self.data_config,
                                            clip_path=str(clip_path))
            preprocessing = preprocess_clip(
                tracking.selected_people,
                confidence_threshold=confidence_threshold,
                missing_joint_strategy=missing_joint_strategy,
                smoothing=bool(self.data_config.get("smoothing", preprocessing_config.get("smoothing", False))),
                smoothing_window=int(self.data_config.get("smoothing_window", 3)),
                scale_epsilon=float(self.data_config.get("scale_epsilon", 1e-6)),
                min_shoulder_width=float(
                    self.data_config.get(
                        "min_shoulder_width", shoulder_config.get("min_width_px", 1.0)
                    )
                ),
                max_interpolation_gap=(
                    int(interpolation_config["max_gap"])
                    if interpolation_config.get("max_gap") is not None
                    else None
                ),
            )
            sequence = self.sequence_builder.extract(preprocessing.coordinates, sequence_metadata)
            if sequence.shape[0] == 0:
                raise ValueError(f"Empty sequence after sampling: {record['clip_id']}")
            if not np.isfinite(sequence).all():
                raise FloatingPointError(f"NaN/Inf in preprocessed sequence: {record['clip_id']}")
            if self.cache_in_memory and not self.return_debug:
                self._sequence_cache[index] = sequence.copy()

        if self.train and self.augmentation.enabled:
            seq_out = self.augmentation(sequence.copy(), self._rng)
        else:
            seq_out = sequence

        sequence_tensor = torch.from_numpy(seq_out.astype(np.float32))
        result: dict[str, object] = {
            "skeleton": sequence_tensor,
            "label": int(record["activity_id"]),
            "clip_id": str(record["clip_id"]),
            "path": str(clip_path),
            "activity": str(record["activity"]),
            "subject_id": str(record.get("subject_id", "unknown")),
            "session_id": str(record.get("session_id", "unknown")),
            **{key: value for key, value in sequence_metadata.items() if key != "clip_index"},
        }
        if self.return_debug:
            result["tracking"] = tracking.debug_dict(str(record["clip_id"]))
            result["preprocessing"] = preprocessing.metadata
            result["valid_mask"] = torch.from_numpy(preprocessing.valid_mask)
        return result


def window_metadata(records: Iterable[dict[str, Any]], sequence: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand an already split clip list; end_frame is inclusive, zero based."""
    sequence = dict(sequence)
    # Compatibility for the original window-study API, which supplied stride only.
    sequence.setdefault("mode", "sliding_window")
    sequence.setdefault("short_clip_policy", "drop")
    builder = SequenceBuilder(sequence)
    windows = []
    rows = list(records)
    for metadata in builder.expand(rows):
        record = rows[int(metadata["clip_index"])]
        windows.append({
            "window_id": metadata["sequence_id"],
            **{key: record[key] for key in ("clip_id", "subject_id", "session_id", "activity", "activity_id")},
            "source_path": record["path"],
            "num_source_frames": metadata["original_num_frames"],
            **metadata,
        })
    return windows


class SkeletonWindowDataset(Dataset[dict[str, object]]):
    """Slice consecutive frames AFTER unchanged full-clip tracking/preprocessing.

    Missing-joint interpolation is retained; temporal resampling is never used.
    Cache full clips once so overlapping windows do not duplicate preprocessing.
    """

    def __init__(self, records, data_config=None, sequence_config=None, return_debug=False, train=False):
        self.records = [dict(row) for row in records]
        self.data_config = data_config or {}
        self.train = train
        augmentation_config = dict(self.data_config.get("augmentation", {}))
        self.augmentation = SkeletonAugmentation(augmentation_config)
        self._rng = np.random.default_rng(int(self.data_config.get("seed", 42)))
        sequence_config = dict(sequence_config or {})
        sequence_config.setdefault("mode", "sliding_window")
        sequence_config.setdefault("short_clip_policy", "drop")
        self.sequence_builder = SequenceBuilder(sequence_config)
        self.windows = window_metadata(self.records, sequence_config)
        if not self.windows:
            raise ValueError("Dataset has no complete 60-frame windows")
        # Keep the indices aligned even when a manifest contains empty clips.
        usable = [row for row in self.records if int(row["num_frames"]) > 0]
        self.clips = SkeletonClipDataset(usable, data_config, {"mode": "full_clip"}, return_debug)
        self._clip_indices = {row["clip_id"]: i for i, row in enumerate(usable)}

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        metadata = self.windows[index]
        clip = self.clips[self._clip_indices[metadata["clip_id"]]]
        if len(clip["skeleton"]) != metadata["num_source_frames"]:
            raise ValueError(f"Manifest/source frame count mismatch: {metadata['clip_id']}")
        start, end = metadata["start_frame"], metadata["end_frame"]
        result = {**clip, **{k: v for k, v in metadata.items() if k != "clip_index"}}
        raw = clip["skeleton"].numpy()
        window_raw = self.sequence_builder.extract(raw, metadata)
        if self.train and self.augmentation.enabled:
            window_raw = self.augmentation(window_raw.copy(), self._rng)
        result["skeleton"] = torch.from_numpy(window_raw.astype(np.float32))
        if "valid_mask" in result:
            result["valid_mask"] = result["valid_mask"][start:end + 1]
        return result

"""Centralised, deterministic construction of 60-frame sequences.

The builder only sees an already tracked and preprocessed ``[T, 25, 2]``
array.  Keeping that boundary explicit prevents a sequence experiment from
silently changing person selection or normalization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np


MODE_ALIASES = {
    "fixed_60": "full_clip_resample",
    "sliding_window": "sliding_window",  # legacy configs resolve from stride
}
MULTI_WINDOW_MODES = {"window_60_stride_60", "window_60_stride_30", "sliding_window"}


def canonical_mode(config: dict[str, Any]) -> str:
    mode = str(config.get("mode", "full_clip_resample"))
    mode = MODE_ALIASES.get(mode, mode)
    if mode == "sliding_window":
        stride = int(config.get("stride", 0))
        if stride in {30, 60}:
            return f"window_60_stride_{stride}"
        # Preserve the old stride-15 study without making it part of Task 3.
        return mode
    return mode


def temporal_positions(length: int, target_length: int = 60) -> np.ndarray:
    if length <= 0 or target_length <= 0:
        raise ValueError("length and target_length must be positive")
    return np.linspace(0.0, float(length - 1), target_length)


def selected_source_indices(length: int, target_length: int = 60) -> np.ndarray:
    """Indices used by the current long-clip implementation (round-to-nearest)."""
    return np.rint(temporal_positions(length, target_length)).astype(np.int64)


def temporal_resample(sequence: np.ndarray, target_length: int = 60) -> np.ndarray:
    """Current exact policy: nearest frames when long, interpolation when short."""
    if sequence.ndim != 3:
        raise ValueError(f"Expected [T,J,2], got {sequence.shape}")
    length = int(sequence.shape[0])
    if length == 0:
        raise ValueError("Cannot resample an empty sequence")
    if length == target_length:
        return sequence.copy()
    positions = temporal_positions(length, target_length)
    if length > target_length:
        return sequence[np.rint(positions).astype(np.int64)]
    if length == 1:
        return np.repeat(sequence, target_length, axis=0)
    lower = np.floor(positions).astype(np.int64)
    upper = np.ceil(positions).astype(np.int64)
    weight = (positions - lower).astype(np.float32).reshape(-1, 1, 1)
    return (sequence[lower] * (1.0 - weight) + sequence[upper] * weight).astype(np.float32)


@dataclass(frozen=True)
class SequenceMetadata:
    source_clip_id: str
    sequence_id: str
    start_frame: int
    end_frame: int
    original_num_frames: int
    window_index: int
    sequence_mode: str
    is_interpolated_short_clip: bool = False


class SequenceBuilder:
    """Describe and extract samples without touching tracking/preprocessing."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        self.mode = canonical_mode(self.config)
        self.length = int(self.config.get("length", self.config.get("window_length", 60)))
        if self.mode in {"uniform_sample", "fixed_window"}:
            self.length = int(self.config.get("target_frames", self.config.get("max_frames", self.length)))
        if self.length != 60 and self.mode not in {"full_clip", "uniform_sample", "fixed_window"}:
            raise ValueError("Task-3 sequence modes require length=60")
        self.short_clip_policy = str(self.config.get("short_clip_policy", "interpolate"))
        if self.short_clip_policy not in {"interpolate", "drop"}:
            raise ValueError("short_clip_policy must be interpolate or drop")
        if self.mode.startswith("window_60_stride_"):
            self.stride = int(self.mode.rsplit("_", 1)[1])
        elif self.mode == "sliding_window":
            self.stride = int(self.config["stride"])
        else:
            self.stride = None
        allowed = {"full_clip", "uniform_sample", "fixed_window", "full_clip_resample", "single_window", "window_60_stride_60",
                   "window_60_stride_30", "sliding_window"}
        if self.mode not in allowed:
            raise ValueError(f"Unknown sequence mode: {self.mode}")
        if self.mode == "sliding_window" and (self.stride is None or self.stride <= 0):
            raise ValueError("sliding_window requires a positive stride")

    @property
    def is_multi_window(self) -> bool:
        return self.mode in MULTI_WINDOW_MODES

    def describe(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        n = int(record["num_frames"])
        if n <= 0:
            return []
        clip_id = str(record["clip_id"])
        spans: list[tuple[int, int, bool]]
        if self.mode in {"full_clip", "full_clip_resample", "uniform_sample"}:
            spans = [(0, n - 1, self.mode == "full_clip_resample" and n < self.length)]
        elif self.mode == "fixed_window":
            start = (n - self.length) // 2 if n > self.length else 0
            spans = [(start, min(start + self.length - 1, n - 1), False)]
        elif self.mode == "single_window":
            if n >= self.length:
                start = (n - self.length) // 2
                spans = [(start, start + self.length - 1, False)]
            elif self.short_clip_policy == "interpolate":
                spans = [(0, n - 1, True)]
            else:
                spans = []
        else:
            if n < self.length:
                spans = [(0, n - 1, True)] if self.short_clip_policy == "interpolate" else []
            else:
                spans = [(start, start + self.length - 1, False)
                         for start in range(0, n - self.length + 1, int(self.stride))]
        result = []
        for index, (start, end, short) in enumerate(spans):
            sequence_id = (clip_id if not self.is_multi_window
                           else f"{clip_id}__start{start:04d}_end{end:04d}")
            metadata = SequenceMetadata(
                source_clip_id=clip_id,
                sequence_id=sequence_id,
                start_frame=start,
                end_frame=end,
                original_num_frames=n,
                window_index=index,
                sequence_mode=self.mode,
                is_interpolated_short_clip=short,
            )
            result.append(asdict(metadata))
        return result

    def expand(self, records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for clip_index, record in enumerate(records):
            for metadata in self.describe(record):
                result.append({**metadata, "clip_index": clip_index})
        return result

    def extract(self, sequence: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
        if sequence.ndim != 3 or sequence.shape[0] != int(metadata["original_num_frames"]):
            raise ValueError("Preprocessed sequence and metadata frame count differ")
        if self.mode == "full_clip":
            return sequence.copy()
        start, end = int(metadata["start_frame"]), int(metadata["end_frame"])
        selected = sequence[start:end + 1]
        if self.mode == "uniform_sample":
            if selected.shape[0] <= self.length:
                return selected.copy()
            indices = np.linspace(0, selected.shape[0] - 1, self.length, dtype=np.int64)
            return selected[indices]
        if self.mode == "fixed_window":
            return selected.copy()
        if self.mode == "full_clip_resample" or bool(metadata["is_interpolated_short_clip"]):
            return temporal_resample(selected, self.length)
        if selected.shape[0] != self.length:
            raise AssertionError("A contiguous sample must contain exactly 60 frames")
        return selected.copy()

"""2D skeleton preprocessing used by the baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .openpose_parser import BODY_25_INDICES, BODY_25_JOINTS


@dataclass
class PreprocessingResult:
    coordinates: np.ndarray  # [T, J, 2]
    valid_mask: np.ndarray  # [T, J], validity before imputation
    metadata: dict[str, object]


def _valid_mask(raw: np.ndarray, threshold: float) -> np.ndarray:
    coords = raw[:, :, :2]
    confidence = raw[:, :, 2]
    return (
        np.isfinite(coords).all(axis=2)
        & np.isfinite(confidence)
        & (confidence >= threshold)
        & ~((coords[:, :, 0] == 0) & (coords[:, :, 1] == 0))
    )


def _missing_runs(valid: np.ndarray) -> list[int]:
    """Return lengths of bounded missing runs for every joint."""
    gaps: list[int] = []
    for joint in range(valid.shape[1]):
        missing = ~valid[:, joint]
        frame = 0
        while frame < len(missing):
            if not missing[frame]:
                frame += 1
                continue
            start = frame
            while frame < len(missing) and missing[frame]:
                frame += 1
            end = frame - 1
            if start > 0 and end < len(missing) - 1:
                gaps.append(end - start + 1)
    return gaps


def _fill_missing(
    coords: np.ndarray,
    valid: np.ndarray,
    strategy: str,
    max_interpolation_gap: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    filled = coords.copy().astype(np.float32)
    filled[~valid] = np.nan
    filled_mask = valid.copy()
    if strategy == "interpolate":
        for joint in range(coords.shape[1]):
            known = np.flatnonzero(valid[:, joint])
            if len(known):
                proposed = np.zeros((coords.shape[0], 2), dtype=np.float32)
                for axis in range(2):
                    proposed[:, axis] = np.interp(
                        np.arange(coords.shape[0]), known, filled[known, joint, axis]
                    )
                for frame in range(coords.shape[0]):
                    if valid[frame, joint]:
                        continue
                    gap_start = frame
                    while gap_start > 0 and not valid[gap_start - 1, joint]:
                        gap_start -= 1
                    gap_end = frame
                    while gap_end + 1 < coords.shape[0] and not valid[gap_end + 1, joint]:
                        gap_end += 1
                    bounded = gap_start > 0 and gap_end < coords.shape[0] - 1
                    gap_length = gap_end - gap_start + 1
                    allowed = max_interpolation_gap is None or gap_length <= max_interpolation_gap
                    if allowed and (bounded or max_interpolation_gap is None):
                        filled[frame, joint] = proposed[frame]
                        filled_mask[frame, joint] = True
    elif strategy == "forward_fill":
        for joint in range(coords.shape[1]):
            last: np.ndarray | None = None
            for frame in range(coords.shape[0]):
                if valid[frame, joint]:
                    last = filled[frame, joint].copy()
                elif last is not None:
                    filled[frame, joint] = last
                    filled_mask[frame, joint] = True
            # Leading gaps cannot be forward-filled; use the first observed
            # value so no NaN reaches the model.
            known = np.flatnonzero(valid[:, joint])
            if len(known):
                filled[: known[0], joint] = filled[known[0], joint]
                filled_mask[: known[0], joint] = True
    elif strategy in {"zero_after_normalization", "mask"}:
        pass
    else:
        raise ValueError(f"Unknown missing_joint_strategy: {strategy}")
    filled[~np.isfinite(filled)] = 0.0
    return filled, filled_mask


def _torso_scale(
    frame: np.ndarray,
    valid: np.ndarray,
    center: np.ndarray,
    shoulder_indices: tuple[int, int],
) -> float:
    points = []
    preferred = [
        BODY_25_INDICES[name]
        for name in ("Neck", "RHip", "LHip", "RShoulder", "LShoulder")
    ]
    for index in preferred:
        if valid[index]:
            distance = float(np.linalg.norm(frame[index] - center))
            if distance > 1e-6:
                points.append(distance)
    if len(points) >= 2:
        return float(np.median(points))
    valid_points = frame[valid]
    if len(valid_points) >= 2:
        return float(max(np.linalg.norm(valid_points.max(axis=0) - valid_points.min(axis=0)), 1.0))
    return 1.0


def preprocess_clip(
    selected_people: Sequence[np.ndarray | None],
    confidence_threshold: float = 0.1,
    missing_joint_strategy: str = "interpolate",
    smoothing: bool = False,
    smoothing_window: int = 3,
    scale_epsilon: float = 1e-6,
    min_shoulder_width: float = 1.0,
    max_interpolation_gap: int | None = None,
) -> PreprocessingResult:
    """Center at MidHip and scale by shoulder width with logged fallbacks.

    OpenPose can produce nearly coincident shoulder coordinates for a frame.
    Treating those coordinates as a real scale would create enormous feature
    values, so a torso-scale fallback is used below a configurable pixel
    threshold.
    """
    frames = len(selected_people)
    if frames == 0:
        raise ValueError("Cannot preprocess an empty clip")
    raw = np.zeros((frames, BODY_25_JOINTS, 3), dtype=np.float32)
    for frame_index, person in enumerate(selected_people):
        if person is not None:
            if person.shape != (BODY_25_JOINTS, 3):
                raise ValueError(f"Expected BODY_25 frame shape {(BODY_25_JOINTS, 3)}, got {person.shape}")
            raw[frame_index] = person
    valid = _valid_mask(raw, confidence_threshold)
    coords, filled_mask = _fill_missing(
        raw[:, :, :2], valid, missing_joint_strategy, max_interpolation_gap
    )
    output = np.zeros_like(coords)
    midhip = BODY_25_INDICES["MidHip"]
    left_shoulder = BODY_25_INDICES["LShoulder"]
    right_shoulder = BODY_25_INDICES["RShoulder"]
    center_valid = 0
    shoulder_scale_frames = 0
    fallback_frames = 0
    invalid_scale_frames = 0
    shoulder_width_threshold = max(float(scale_epsilon), float(min_shoulder_width))

    for frame in range(frames):
        if valid[frame, midhip] or missing_joint_strategy in {"interpolate", "forward_fill"}:
            center = coords[frame, midhip].copy()
            center_valid += int(valid[frame, midhip])
        else:
            center = np.zeros(2, dtype=np.float32)

        centered = coords[frame] - center
        if valid[frame, left_shoulder] and valid[frame, right_shoulder]:
            scale = float(np.linalg.norm(centered[left_shoulder] - centered[right_shoulder]))
            if scale > shoulder_width_threshold:
                shoulder_scale_frames += 1
            else:
                scale = _torso_scale(centered, valid[frame], np.zeros(2), (left_shoulder, right_shoulder))
                fallback_frames += 1
                invalid_scale_frames += 1
        else:
            scale = _torso_scale(centered, valid[frame], np.zeros(2), (left_shoulder, right_shoulder))
            fallback_frames += 1
        if scale <= scale_epsilon or not np.isfinite(scale):
            scale = 1.0
            invalid_scale_frames += 1
        output[frame] = centered / scale

    if missing_joint_strategy in {"zero_after_normalization", "mask"}:
        output[~valid] = 0.0

    if smoothing:
        window = max(int(smoothing_window), 1)
        if window > 1:
            kernel = np.ones(window, dtype=np.float32) / window
            for joint in range(BODY_25_JOINTS):
                for axis in range(2):
                    output[:, joint, axis] = np.convolve(
                        output[:, joint, axis], kernel, mode="same"
                    )

    metadata = {
        "confidence_threshold": confidence_threshold,
        "missing_joint_strategy": missing_joint_strategy,
        "smoothing": smoothing,
        "min_shoulder_width": min_shoulder_width,
        "shoulder_width_threshold": shoulder_width_threshold,
        "center_valid_frames": center_valid,
        "invalid_midhip_frames": int((~valid[:, midhip]).sum()),
        "invalid_midhip_percentage": float(100.0 * (~valid[:, midhip]).mean()),
        "invalid_shoulder_pair_frames": int((~(valid[:, left_shoulder] & valid[:, right_shoulder])).sum()),
        "invalid_shoulder_pair_percentage": float(
            100.0 * (~(valid[:, left_shoulder] & valid[:, right_shoulder])).mean()
        ),
        "shoulder_scale_frames": shoulder_scale_frames,
        "scale_fallback_frames": fallback_frames,
        "scale_fallback_percentage": float(100.0 * fallback_frames / frames),
        "invalid_scale_frames": invalid_scale_frames,
        "missing_joint_percentage": float(100.0 * (~valid).mean()),
        "interpolated_joint_percentage": float(100.0 * (filled_mask & ~valid).mean()),
        "interpolation_max_gap": max(_missing_runs(valid), default=0),
        "interpolation_median_gap": float(np.median(_missing_runs(valid))) if _missing_runs(valid) else 0.0,
        "max_interpolation_gap_config": max_interpolation_gap,
    }
    return PreprocessingResult(output.astype(np.float32), valid, metadata)

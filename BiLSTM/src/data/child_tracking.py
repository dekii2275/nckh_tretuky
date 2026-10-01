"""Deterministic child/person track selection for multi-person OpenPose frames."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .openpose_parser import BODY_25_INDICES


@dataclass
class PersonFeatures:
    center: np.ndarray
    torso: np.ndarray
    diagonal: float
    mean_confidence: float
    valid_count: int


@dataclass
class TrackingResult:
    selected_indices: list[int]
    selected_people: list[np.ndarray | None]
    failures: list[dict[str, object]]
    strategy: str

    def debug_dict(self, clip_id: str) -> dict[str, object]:
        return {
            "clip_id": clip_id,
            "selected_person_per_frame": self.selected_indices,
            "tracking_failures": self.failures,
            "strategy": self.strategy,
        }


def _valid_mask(person: np.ndarray, threshold: float) -> np.ndarray:
    coords = person[:, :2]
    confidence = person[:, 2]
    return (
        np.isfinite(coords).all(axis=1)
        & np.isfinite(confidence)
        & (confidence >= threshold)
        & ~((coords[:, 0] == 0) & (coords[:, 1] == 0))
    )


def person_features(person: np.ndarray, confidence_threshold: float = 0.1) -> PersonFeatures | None:
    valid = _valid_mask(person, confidence_threshold)
    if not valid.any():
        return None
    coords = person[:, :2]
    points = coords[valid]
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = (low + high) / 2.0
    diagonal = float(np.linalg.norm(high - low))
    diagonal = max(diagonal, 1.0)
    torso_indices = [
        BODY_25_INDICES[name]
        for name in ("MidHip", "Neck", "RHip", "LHip", "RShoulder", "LShoulder")
    ]
    torso_valid = valid[torso_indices]
    torso_points = coords[np.asarray(torso_indices)[torso_valid]]
    torso = torso_points.mean(axis=0) if len(torso_points) else center
    confidence = person[:, 2][valid]
    return PersonFeatures(center, torso, diagonal, float(confidence.mean()), int(valid.sum()))


def _initial_score(features: PersonFeatures, prefer_smaller: bool = True) -> float:
    # In this dataset an adult therapist is often the largest skeleton. A
    # modest size prior is therefore a useful, configurable child proxy. It is
    # not ground-truth identity and is deliberately documented as a limitation.
    quality = features.mean_confidence * math.log1p(features.valid_count)
    if prefer_smaller:
        return quality / math.sqrt(max(features.diagonal, 1.0))
    return quality


def _transition_cost(previous: PersonFeatures, current: PersonFeatures) -> float:
    reference_scale = max(previous.diagonal, current.diagonal, 1.0)
    center_distance = np.linalg.norm(previous.center - current.center) / reference_scale
    torso_distance = np.linalg.norm(previous.torso - current.torso) / reference_scale
    size_change = abs(math.log(max(current.diagonal, 1.0) / max(previous.diagonal, 1.0)))
    confidence_penalty = 1.0 - current.mean_confidence
    return (
        0.45 * float(center_distance)
        + 0.35 * float(torso_distance)
        + 0.15 * float(size_change)
        + 0.05 * confidence_penalty
    )


def select_child_track(
    frames: Sequence[Sequence[np.ndarray]],
    confidence_threshold: float = 0.1,
    strategy: str = "temporal_nearest",
    prefer_smaller: bool = True,
) -> TrackingResult:
    """Select one temporally consistent person per frame.

    The source MMASD OpenPose JSON has ``person_id=[-1]`` and no child/therapist
    annotation. Therefore this function returns a tracked person *candidate*;
    it cannot prove that the candidate is the child. The limitation is exposed
    in the README and the strategy is configurable.
    """
    if strategy not in {"temporal_nearest", "highest_confidence"}:
        raise ValueError(f"Unknown tracking strategy: {strategy}")
    selected_indices: list[int] = []
    selected_people: list[np.ndarray | None] = []
    failures: list[dict[str, object]] = []
    previous: PersonFeatures | None = None

    for frame_index, people in enumerate(frames):
        candidates: list[tuple[int, np.ndarray, PersonFeatures]] = []
        for person_index, person in enumerate(people):
            features = person_features(person, confidence_threshold)
            if features is not None:
                candidates.append((person_index, person, features))
        if not candidates:
            selected_indices.append(-1)
            selected_people.append(None)
            failures.append({"frame_index": frame_index, "reason": "no_valid_person"})
            continue

        if previous is None:
            chosen = max(
                candidates,
                key=lambda item: (_initial_score(item[2], prefer_smaller), -item[0]),
            )
        elif strategy == "highest_confidence":
            chosen = max(candidates, key=lambda item: (item[2].mean_confidence, item[2].valid_count, -item[0]))
        else:
            chosen = min(candidates, key=lambda item: (_transition_cost(previous, item[2]), item[0]))
        selected_indices.append(chosen[0])
        selected_people.append(chosen[1].copy())
        previous = chosen[2]

    return TrackingResult(selected_indices, selected_people, failures, strategy)

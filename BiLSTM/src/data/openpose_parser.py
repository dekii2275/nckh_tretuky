"""Parser for OpenPose BODY_25 JSON output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


BODY_25_NAMES = (
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "MidHip",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
    "LBigToe",
    "LSmallToe",
    "LHeel",
    "RBigToe",
    "RSmallToe",
    "RHeel",
)
BODY_25_INDICES = {name: index for index, name in enumerate(BODY_25_NAMES)}
BODY_25_JOINTS = len(BODY_25_NAMES)


class OpenPoseFormatError(ValueError):
    """Raised when a JSON frame is not a supported BODY_25 frame."""


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise OpenPoseFormatError(f"Expected JSON object: {path}")
    return value


def parse_frame(path: str | Path, strict: bool = True) -> list[np.ndarray]:
    """Return people as arrays shaped ``[25, 3]`` containing x, y, confidence."""
    payload = _read_json(path)
    people = payload.get("people", [])
    if not isinstance(people, list):
        raise OpenPoseFormatError(f"'people' is not a list: {path}")
    parsed: list[np.ndarray] = []
    for person_index, person in enumerate(people):
        values = person.get("pose_keypoints_2d", []) if isinstance(person, dict) else []
        if len(values) != BODY_25_JOINTS * 3:
            if strict:
                raise OpenPoseFormatError(
                    f"Person {person_index} in {path} has {len(values)} values; "
                    f"expected {BODY_25_JOINTS * 3}"
                )
            continue
        array = np.asarray(values, dtype=np.float32).reshape(BODY_25_JOINTS, 3)
        parsed.append(array)
    return parsed


def parse_clip(clip_path: str | Path, strict: bool = True) -> tuple[list[np.ndarray], list[Path]]:
    from .discover import frame_paths

    paths = frame_paths(clip_path)
    frames = [parse_frame(path, strict=strict) for path in paths]
    return frames, paths

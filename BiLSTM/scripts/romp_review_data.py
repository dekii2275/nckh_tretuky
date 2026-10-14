"""Read ROMP review coordinates without renumbering people or hiding gaps.

SMPL24 order follows Arthur151/ROMP simple_romp/README.md. ROMP71 keeps
these 24 joints first. The coordinate-only MMASD export does not provide a
metric calibration or SMPL pose/shape parameters.
"""
from __future__ import annotations

import re
from zipfile import BadZipFile
from pathlib import Path

import numpy as np

# (SMPL24 index, OpenPose BODY25 index), anatomically corresponding joints.
MATCH_JOINTS = ((16, 5), (17, 2), (18, 6), (19, 3), (20, 7), (21, 4),
                (1, 12), (2, 9), (4, 13), (5, 10), (7, 14), (8, 11))


def frame_id(path: Path) -> int:
    match = re.search(r"(?:^|_)(\d+)(?:_keypoints)?$", path.stem)
    if not match:
        raise ValueError(f"Cannot identify source frame: {path.name}")
    return int(match.group(1))


def read_romp_frames(directory: Path):
    frames, errors = {}, []
    for path in sorted(directory.glob("*.npz")):
        index = frame_id(path)
        if index in frames:
            raise ValueError(f"Duplicate ROMP frame {index}: {path}")
        try:
            with np.load(path, allow_pickle=False) as data:
                coordinates = data["coordinates"]
            if coordinates.ndim != 3 or coordinates.shape[1] not in (24, 71) or coordinates.shape[2] != 3:
                raise ValueError(f"Unsupported ROMP coordinates {coordinates.shape}")
            # Keep candidate slots and partial poses. JSON null marks missing joints.
            people = []
            for person in coordinates[:, :24]:
                people.append([
                    point.astype(float).tolist() if np.isfinite(point).all() else None
                    for point in person
                ])
            frames[index] = people
        except (OSError, ValueError, KeyError, EOFError, BadZipFile) as error:
            frames[index] = []
            errors.append({"frame_id": index, "file": path.name, "reason": str(error)})
    return frames, errors


def match_selected_person(openpose_person, romp_people, threshold=0.1,
                          max_distance=0.18, ambiguity_margin=0.025):
    """Match common joints in image XY; return unmatched/ambiguous explicitly.

    Scores are median pixel distances divided by the ROMP image bounding-box
    diagonal. They are association heuristics, not calibrated probabilities.
    No claim about metric Z or human child identity is made here.
    """
    if openpose_person is None:
        return {"index": -1, "status": "unmatched", "score": None}
    scores = []
    for index, person in enumerate(romp_people):
        distances, points = [], []
        for smpl, body in MATCH_JOINTS:
            point = person[smpl]
            observed = openpose_person[body]
            if point is None or not np.isfinite(point).all():
                continue
            if not np.isfinite(observed).all() or observed[2] < threshold or not observed[:2].any():
                continue
            if not np.asarray(point).any():
                continue
            points.append(point[:2])
            distances.append(float(np.linalg.norm(observed[:2] - np.asarray(point[:2]))))
        if len(distances) < 4:
            continue
        points = np.asarray(points)
        diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
        if diagonal < 10:
            continue
        scores.append((float(np.median(distances)) / diagonal, index))
    scores.sort()
    if not scores or scores[0][0] > max_distance:
        return {"index": -1, "status": "unmatched", "score": scores[0][0] if scores else None}
    if len(scores) > 1 and scores[1][0] - scores[0][0] < ambiguity_margin:
        return {"index": -1, "status": "ambiguous", "score": scores[0][0]}
    return {"index": scores[0][1], "status": "matched", "score": scores[0][0]}

"""ROMP-guided person selection for OpenPose BODY_25 frames.

Uses ROMP 2D coordinate bounding boxes to identify the target person in each
frame, then maps to the closest OpenPose person by centroid distance.  This
provides a more stable tracking signal than heuristic-only approaches because
ROMP's person detection is temporally consistent per-clip.

The target person is selected from ROMP detections as the **smallest** skeleton
(child proxy) consistently present across the clip.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .child_tracking import TrackingResult, person_features
from .openpose_parser import BODY_25_JOINTS


@dataclass
class ROMPGuidedTrackingResult(TrackingResult):
    romp_coverage: float = 0.0
    romp_person_index: int = -1
    match_distances: list[float] | None = None


def _romp_clip_dir(openpose_clip_path: Path) -> Path | None:
    """Resolve the ROMP 2D directory corresponding to an OpenPose clip.

    OpenPose layout:  .../2D_openpose/output/<activity>/<clip_id>/
    ROMP 2D layout:   .../ROMP_2D_Coordinates/<activity>/<clip_id>/
    """
    # Walk up from the openpose clip dir to find the data_MMASD root
    parts = openpose_clip_path.parts
    try:
        idx = parts.index("2D_openpose")
    except ValueError:
        return None
    # data_MMASD / 2D_skeleton / 2D_openpose / output / <activity> / <clip_id>
    data_root = Path(*parts[:idx])  # .../2D_skeleton
    activity = parts[idx + 2] if len(parts) > idx + 2 else None
    clip_id = parts[idx + 3] if len(parts) > idx + 3 else None
    if activity is None or clip_id is None:
        return None
    romp_dir = data_root / "ROMP_2D_Coordinates" / activity / clip_id
    return romp_dir if romp_dir.is_dir() else None


def load_romp_bboxes(romp_dir: Path) -> list[np.ndarray | None]:
    """Load per-frame ROMP 2D coordinates and compute bounding boxes.

    Returns a list of arrays shaped ``[N_persons, 4]`` (x_min, y_min, x_max, y_max)
    per frame, or None for missing frames.
    """
    frame_files = sorted(romp_dir.glob("*.npz"))
    if not frame_files:
        return []
    result: list[np.ndarray | None] = []
    for path in frame_files:
        try:
            data = np.load(path, allow_pickle=True)
            coords = data["coordinates"]  # [N_persons, 24, 2]
            if coords.ndim != 3 or coords.shape[2] != 2:
                result.append(None)
                continue
            bboxes = np.zeros((coords.shape[0], 4), dtype=np.float32)
            for p in range(coords.shape[0]):
                valid = np.isfinite(coords[p]).all(axis=1) & (coords[p] != 0).any(axis=1)
                if valid.any():
                    pts = coords[p][valid]
                    bboxes[p] = [pts[:, 0].min(), pts[:, 1].min(),
                                 pts[:, 0].max(), pts[:, 1].max()]
                else:
                    bboxes[p] = [0, 0, 0, 0]
            result.append(bboxes)
        except Exception:
            result.append(None)
    return result


def _bbox_center(bbox: np.ndarray) -> np.ndarray:
    return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])


def _bbox_diagonal(bbox: np.ndarray) -> float:
    return float(np.linalg.norm(bbox[2:4] - bbox[0:2]))


def select_romp_target(
    romp_bboxes: list[np.ndarray | None],
    prefer_smaller: bool = True,
) -> int:
    """Select the ROMP person index most likely to be the child.

    Strategy: pick the person with the highest coverage across frames that
    also has a consistently small bounding box (child proxy).
    """
    if not romp_bboxes:
        return -1

    # Collect statistics per person index
    max_persons = max(
        (bb.shape[0] for bb in romp_bboxes if bb is not None), default=0
    )
    if max_persons == 0:
        return -1

    # For each person index, count frames present and median diagonal
    coverage: dict[int, int] = {}
    diagonals: dict[int, list[float]] = {}
    for bb in romp_bboxes:
        if bb is None:
            continue
        for p in range(bb.shape[0]):
            diag = _bbox_diagonal(bb[p])
            if diag > 10:  # valid detection
                coverage[p] = coverage.get(p, 0) + 1
                diagonals.setdefault(p, []).append(diag)

    if not coverage:
        return -1

    total_frames = sum(1 for bb in romp_bboxes if bb is not None)
    min_coverage = 0.3  # must be present in at least 30% of frames

    candidates = [
        p for p, c in coverage.items()
        if c / max(total_frames, 1) >= min_coverage
    ]
    if not candidates:
        # Fall back to the person with highest coverage
        return max(coverage, key=lambda p: coverage[p])

    if prefer_smaller:
        # Among candidates with sufficient coverage, pick the smallest
        return min(candidates, key=lambda p: float(np.median(diagonals[p])))
    else:
        # Pick highest coverage
        return max(candidates, key=lambda p: coverage[p])


def match_romp_to_openpose(
    romp_bbox: np.ndarray,
    romp_person_idx: int,
    openpose_people: Sequence[np.ndarray],
    confidence_threshold: float = 0.1,
) -> int:
    """Find the OpenPose person closest to the selected ROMP person."""
    if romp_person_idx < 0 or romp_person_idx >= romp_bbox.shape[0]:
        return -1

    target_center = _bbox_center(romp_bbox[romp_person_idx])
    target_diag = _bbox_diagonal(romp_bbox[romp_person_idx])
    if target_diag < 10:
        return -1

    best_idx = -1
    best_dist = float("inf")

    for op_idx, person in enumerate(openpose_people):
        features = person_features(person, confidence_threshold)
        if features is None:
            continue
        dist = float(np.linalg.norm(features.center - target_center))
        # Normalize by target diagonal for scale invariance
        normalized_dist = dist / max(target_diag, 1.0)
        if normalized_dist < best_dist:
            best_dist = normalized_dist
            best_idx = op_idx

    return best_idx


def select_romp_guided_track(
    frames: Sequence[Sequence[np.ndarray]],
    openpose_clip_path: str | Path,
    confidence_threshold: float = 0.1,
    prefer_smaller: bool = True,
    max_match_distance: float = 0.5,
) -> ROMPGuidedTrackingResult:
    """Select OpenPose person per frame guided by ROMP 2D detections.

    Falls back to the legacy heuristic tracker when ROMP data is unavailable.
    """
    clip_path = Path(openpose_clip_path)
    romp_dir = _romp_clip_dir(clip_path)

    if romp_dir is None or not romp_dir.exists():
        # Fall back to legacy tracking
        from .child_tracking import select_child_track
        legacy = select_child_track(
            frames, confidence_threshold=confidence_threshold,
            strategy="temporal_nearest", prefer_smaller=prefer_smaller,
        )
        return ROMPGuidedTrackingResult(
            selected_indices=legacy.selected_indices,
            selected_people=legacy.selected_people,
            failures=legacy.failures,
            strategy="romp_guided_fallback_legacy",
            romp_coverage=0.0,
            romp_person_index=-1,
        )

    romp_bboxes = load_romp_bboxes(romp_dir)
    if not romp_bboxes or len(romp_bboxes) != len(frames):
        from .child_tracking import select_child_track
        legacy = select_child_track(
            frames, confidence_threshold=confidence_threshold,
            strategy="temporal_nearest", prefer_smaller=prefer_smaller,
        )
        return ROMPGuidedTrackingResult(
            selected_indices=legacy.selected_indices,
            selected_people=legacy.selected_people,
            failures=legacy.failures,
            strategy="romp_guided_fallback_frame_mismatch",
            romp_coverage=0.0,
            romp_person_index=-1,
        )

    target_person = select_romp_target(romp_bboxes, prefer_smaller=prefer_smaller)
    if target_person < 0:
        from .child_tracking import select_child_track
        legacy = select_child_track(
            frames, confidence_threshold=confidence_threshold,
            strategy="temporal_nearest", prefer_smaller=prefer_smaller,
        )
        return ROMPGuidedTrackingResult(
            selected_indices=legacy.selected_indices,
            selected_people=legacy.selected_people,
            failures=legacy.failures,
            strategy="romp_guided_fallback_no_target",
            romp_coverage=0.0,
            romp_person_index=-1,
        )

    selected_indices: list[int] = []
    selected_people: list[np.ndarray | None] = []
    failures: list[dict[str, object]] = []
    match_distances: list[float] = []
    romp_hits = 0

    # Track previous valid selection for temporal continuity fallback
    prev_center: np.ndarray | None = None

    for frame_idx, (people, romp_bb) in enumerate(zip(frames, romp_bboxes)):
        if romp_bb is None or target_person >= romp_bb.shape[0]:
            # ROMP missing this frame — use temporal continuity
            if prev_center is not None and people:
                best_idx = -1
                best_dist = float("inf")
                for op_idx, person in enumerate(people):
                    features = person_features(person, confidence_threshold)
                    if features is None:
                        continue
                    dist = float(np.linalg.norm(features.center - prev_center))
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = op_idx
                if best_idx >= 0:
                    selected_indices.append(best_idx)
                    selected_people.append(people[best_idx].copy())
                    match_distances.append(best_dist)
                    prev_center = person_features(people[best_idx], confidence_threshold).center
                    continue

            selected_indices.append(-1)
            selected_people.append(None)
            failures.append({"frame_index": frame_idx, "reason": "romp_missing"})
            match_distances.append(float("inf"))
            continue

        op_idx = match_romp_to_openpose(
            romp_bb, target_person, people, confidence_threshold
        )

        if op_idx < 0:
            selected_indices.append(-1)
            selected_people.append(None)
            failures.append({"frame_index": frame_idx, "reason": "no_openpose_match"})
            match_distances.append(float("inf"))
            continue

        # Verify match distance
        target_center = _bbox_center(romp_bb[target_person])
        target_diag = _bbox_diagonal(romp_bb[target_person])
        op_features = person_features(people[op_idx], confidence_threshold)
        dist = float(np.linalg.norm(op_features.center - target_center)) / max(target_diag, 1.0)

        if dist > max_match_distance:
            # Match too far — likely wrong person
            selected_indices.append(-1)
            selected_people.append(None)
            failures.append({"frame_index": frame_idx, "reason": "match_too_far",
                             "distance": dist})
            match_distances.append(dist)
            continue

        selected_indices.append(op_idx)
        selected_people.append(people[op_idx].copy())
        match_distances.append(dist)
        romp_hits += 1
        prev_center = op_features.center

    coverage = romp_hits / max(len(frames), 1)

    return ROMPGuidedTrackingResult(
        selected_indices=selected_indices,
        selected_people=selected_people,
        failures=failures,
        strategy="romp_guided",
        romp_coverage=coverage,
        romp_person_index=target_person,
        match_distances=match_distances,
    )

"""Quality-filtered, gated multi-person tracklets for OpenPose BODY_25.

This module tracks anonymous person candidates.  It does not classify a track
as child or therapist because MMASD OpenPose provides no verified role labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .child_tracking import TrackingResult
from .openpose_parser import BODY_25_INDICES, BODY_25_JOINTS


TORSO_ANCHOR_NAMES = ("Neck", "MidHip", "LShoulder", "RShoulder", "LHip", "RHip")
TORSO_ANCHORS = np.asarray([BODY_25_INDICES[name] for name in TORSO_ANCHOR_NAMES])


@dataclass(frozen=True)
class CandidateQualityConfig:
    min_valid_joints: int = 10
    min_torso_anchors: int = 4
    min_diagonal_px: float = 50.0


@dataclass(frozen=True)
class GatingConfig:
    max_normalized_centroid_distance: float = 0.50
    max_scale_ratio: float = 1.80
    max_pose_distance: float = 0.35
    min_shared_joints: int = 6


@dataclass(frozen=True)
class MatchingWeights:
    center: float = 0.40
    pose: float = 0.35
    scale: float = 0.20
    confidence: float = 0.05


@dataclass(frozen=True)
class TargetSelectionConfig:
    coverage_weight: float = 0.45
    valid_joints_weight: float = 0.20
    confidence_weight: float = 0.15
    continuity_weight: float = 0.10
    scale_stability_weight: float = 0.05
    low_fragmentation_weight: float = 0.05
    use_size_prior: bool = False
    size_prior_weight: float = 0.02


@dataclass
class TrackCandidate:
    raw_index: int
    person: np.ndarray
    valid_mask: np.ndarray
    center: np.ndarray
    torso: np.ndarray
    diagonal: float
    mean_confidence: float
    valid_count: int
    torso_anchor_count: int


@dataclass
class MatchMetrics:
    cost: float
    center_distance: float
    pose_distance: float
    scale_change: float
    scale_ratio: float
    shared_joints: int


@dataclass
class TrackObservation:
    frame_index: int
    candidate: TrackCandidate
    match: MatchMetrics | None


@dataclass
class Tracklet:
    track_id: int
    start_frame: int
    observations: dict[int, TrackObservation] = field(default_factory=dict)
    last_candidate: TrackCandidate | None = None
    previous_candidate: TrackCandidate | None = None
    last_frame: int = -1
    previous_frame: int = -1
    lost_frames: int = 0
    lost_events: int = 0
    terminated: bool = False

    @property
    def hits(self) -> int:
        return len(self.observations)

    def add(self, frame_index: int, candidate: TrackCandidate, match: MatchMetrics | None) -> None:
        if self.lost_frames:
            self.lost_events += 1
        self.previous_candidate = self.last_candidate
        self.previous_frame = self.last_frame
        self.last_candidate = candidate
        self.last_frame = frame_index
        self.lost_frames = 0
        self.observations[frame_index] = TrackObservation(frame_index, candidate, match)

    def mark_lost(self, max_lost_frames: int) -> None:
        self.lost_frames += 1
        if self.lost_frames > max_lost_frames:
            self.terminated = True

    def predicted_center(self, frame_index: int) -> np.ndarray:
        assert self.last_candidate is not None
        if self.previous_candidate is None or self.previous_frame < 0:
            return self.last_candidate.center
        observed_gap = max(self.last_frame - self.previous_frame, 1)
        velocity = (self.last_candidate.center - self.previous_candidate.center) / observed_gap
        return self.last_candidate.center + velocity * max(frame_index - self.last_frame, 1)


@dataclass
class TrackletTrackingResult(TrackingResult):
    target_track_id: int = -1
    tracks: list[Tracklet] = field(default_factory=list)
    rejected_candidates_per_frame: list[int] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def debug_dict(self, clip_id: str) -> dict[str, object]:
        result = super().debug_dict(clip_id)
        result.update(
            {
                "target_track_id": self.target_track_id,
                "rejected_candidates_per_frame": self.rejected_candidates_per_frame,
                "tracklet_config": self.config,
                "tracks": [track_summary(track, len(self.selected_people), self.config) for track in self.tracks],
            }
        )
        return result


def _valid_mask(person: np.ndarray, threshold: float) -> np.ndarray:
    if person.shape != (BODY_25_JOINTS, 3):
        return np.zeros(BODY_25_JOINTS, dtype=bool)
    coords = person[:, :2]
    confidence = person[:, 2]
    return (
        np.isfinite(coords).all(axis=1)
        & np.isfinite(confidence)
        & (confidence >= threshold)
        & ~((coords[:, 0] == 0) & (coords[:, 1] == 0))
    )


def candidate_from_person(
    person: np.ndarray,
    raw_index: int,
    confidence_threshold: float,
    quality: CandidateQualityConfig,
) -> tuple[TrackCandidate | None, str | None]:
    """Build a candidate or return an explicit quality-rejection reason."""
    valid = _valid_mask(person, confidence_threshold)
    valid_count = int(valid.sum())
    if valid_count < quality.min_valid_joints:
        return None, "insufficient_valid_joints"
    torso_count = int(valid[TORSO_ANCHORS].sum())
    if torso_count < quality.min_torso_anchors:
        return None, "insufficient_torso_anchors"
    points = person[valid, :2]
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2.0
    diagonal = float(np.linalg.norm(high - low))
    torso_points = person[TORSO_ANCHORS[valid[TORSO_ANCHORS]], :2]
    torso = torso_points.mean(axis=0)
    confidence = float(person[valid, 2].mean())
    if not (
        np.isfinite(center).all()
        and np.isfinite(torso).all()
        and np.isfinite(diagonal)
        and np.isfinite(confidence)
    ):
        return None, "non_finite_geometry"
    if diagonal < quality.min_diagonal_px:
        return None, "diagonal_below_minimum"
    return (
        TrackCandidate(
            raw_index=raw_index,
            person=person,
            valid_mask=valid,
            center=center,
            torso=torso,
            diagonal=diagonal,
            mean_confidence=confidence,
            valid_count=valid_count,
            torso_anchor_count=torso_count,
        ),
        None,
    )


def match_metrics(
    track: Tracklet,
    candidate: TrackCandidate,
    frame_index: int,
    gating: GatingConfig,
    weights: MatchingWeights,
) -> MatchMetrics | None:
    """Return normalized match metrics, or None when any hard gate fails."""
    previous = track.last_candidate
    assert previous is not None
    gap = max(frame_index - track.last_frame, 1)
    reference_scale = max(previous.diagonal, candidate.diagonal, 1.0)
    center_distance = float(np.linalg.norm(track.predicted_center(frame_index) - candidate.center) / reference_scale)
    scale_ratio = max(previous.diagonal, candidate.diagonal) / max(min(previous.diagonal, candidate.diagonal), 1.0)
    scale_change = abs(math.log(max(candidate.diagonal, 1.0) / max(previous.diagonal, 1.0)))
    shared = previous.valid_mask & candidate.valid_mask
    shared_count = int(shared.sum())
    if shared_count < gating.min_shared_joints:
        return None
    previous_pose = (previous.person[shared, :2] - previous.torso) / max(previous.diagonal, 1.0)
    current_pose = (candidate.person[shared, :2] - candidate.torso) / max(candidate.diagonal, 1.0)
    pose_distance = float(np.linalg.norm(previous_pose - current_pose, axis=1).mean())
    gap_relaxation = math.sqrt(float(gap))
    if (
        center_distance > gating.max_normalized_centroid_distance * gap_relaxation
        or scale_ratio > gating.max_scale_ratio
        or pose_distance > gating.max_pose_distance * gap_relaxation
    ):
        return None
    confidence_penalty = 1.0 - candidate.mean_confidence
    cost = (
        weights.center * center_distance
        + weights.pose * pose_distance
        + weights.scale * scale_change
        + weights.confidence * confidence_penalty
    )
    return MatchMetrics(cost, center_distance, pose_distance, scale_change, scale_ratio, shared_count)


def _quality_config(config: dict[str, Any]) -> CandidateQualityConfig:
    values = dict(config.get("candidate_filter", {}))
    return CandidateQualityConfig(
        min_valid_joints=int(values.get("min_valid_joints", 10)),
        min_torso_anchors=int(values.get("min_torso_anchors", 4)),
        min_diagonal_px=float(values.get("min_diagonal_px", 50.0)),
    )


def _gating_config(config: dict[str, Any]) -> GatingConfig:
    values = dict(config.get("gating", {}))
    return GatingConfig(
        max_normalized_centroid_distance=float(values.get("max_normalized_centroid_distance", 0.50)),
        max_scale_ratio=float(values.get("max_scale_ratio", 1.80)),
        max_pose_distance=float(values.get("max_pose_distance", 0.35)),
        min_shared_joints=int(values.get("min_shared_joints", 6)),
    )


def _matching_weights(config: dict[str, Any]) -> MatchingWeights:
    values = dict(config.get("matching", {}).get("weights", {}))
    result = MatchingWeights(
        center=float(values.get("center", 0.40)),
        pose=float(values.get("pose", 0.35)),
        scale=float(values.get("scale", 0.20)),
        confidence=float(values.get("confidence", 0.05)),
    )
    if min(result.center, result.pose, result.scale, result.confidence) < 0:
        raise ValueError("tracking.matching.weights must be non-negative")
    if result.center + result.pose + result.scale + result.confidence <= 0:
        raise ValueError("At least one tracking matching weight must be positive")
    return result


def _target_config(config: dict[str, Any]) -> TargetSelectionConfig:
    values = dict(config.get("target_selection", {}))
    return TargetSelectionConfig(
        coverage_weight=float(values.get("coverage_weight", 0.45)),
        valid_joints_weight=float(values.get("valid_joints_weight", 0.20)),
        confidence_weight=float(values.get("confidence_weight", 0.15)),
        continuity_weight=float(values.get("continuity_weight", 0.10)),
        scale_stability_weight=float(values.get("scale_stability_weight", 0.05)),
        low_fragmentation_weight=float(values.get("low_fragmentation_weight", 0.05)),
        use_size_prior=bool(values.get("use_size_prior", False)),
        size_prior_weight=float(values.get("size_prior_weight", 0.02)),
    )


def validate_config(config: dict[str, Any]) -> None:
    quality, gating, target = _quality_config(config), _gating_config(config), _target_config(config)
    if not 1 <= quality.min_valid_joints <= BODY_25_JOINTS:
        raise ValueError("min_valid_joints must be in [1,25]")
    if not 1 <= quality.min_torso_anchors <= len(TORSO_ANCHORS):
        raise ValueError("min_torso_anchors must be in [1,6]")
    if quality.min_diagonal_px <= 0:
        raise ValueError("min_diagonal_px must be positive")
    if min(gating.max_normalized_centroid_distance, gating.max_scale_ratio, gating.max_pose_distance) <= 0:
        raise ValueError("tracking gates must be positive")
    if not 1 <= gating.min_shared_joints <= BODY_25_JOINTS:
        raise ValueError("min_shared_joints must be in [1,25]")
    if int(config.get("tracks", {}).get("max_lost_frames", 3)) < 0:
        raise ValueError("max_lost_frames must be non-negative")
    target_weights = (
        target.coverage_weight,
        target.valid_joints_weight,
        target.confidence_weight,
        target.continuity_weight,
        target.scale_stability_weight,
        target.low_fragmentation_weight,
        target.size_prior_weight,
    )
    if min(target_weights) < 0:
        raise ValueError("target-selection weights must be non-negative")


def build_tracklets(
    frames: Sequence[Sequence[np.ndarray]],
    confidence_threshold: float,
    config: dict[str, Any],
) -> tuple[list[Tracklet], list[int], list[dict[str, int]]]:
    """Build anonymous one-to-one tracklets before selecting any target track."""
    validate_config(config)
    quality, gating, weights = _quality_config(config), _gating_config(config), _matching_weights(config)
    max_lost = int(config.get("tracks", {}).get("max_lost_frames", 3))
    tracks: list[Tracklet] = []
    rejected_counts: list[int] = []
    rejection_reasons: list[dict[str, int]] = []
    next_track_id = 0
    for frame_index, people in enumerate(frames):
        candidates: list[TrackCandidate] = []
        reasons: dict[str, int] = {}
        for raw_index, person in enumerate(people):
            candidate, reason = candidate_from_person(person, raw_index, confidence_threshold, quality)
            if candidate is None:
                assert reason is not None
                reasons[reason] = reasons.get(reason, 0) + 1
            else:
                candidates.append(candidate)
        rejected_counts.append(sum(reasons.values()))
        rejection_reasons.append(reasons)
        active = [track for track in tracks if not track.terminated and track.lost_frames <= max_lost]
        edges: list[tuple[float, int, int, MatchMetrics]] = []
        for track_position, track in enumerate(active):
            for candidate_position, candidate in enumerate(candidates):
                metrics = match_metrics(track, candidate, frame_index, gating, weights)
                if metrics is not None:
                    edges.append((metrics.cost, track_position, candidate_position, metrics))
        edges.sort(key=lambda item: (item[0], active[item[1]].track_id, candidates[item[2]].raw_index))
        matched_tracks: set[int] = set()
        matched_candidates: set[int] = set()
        for _, track_position, candidate_position, metrics in edges:
            track = active[track_position]
            if track.track_id in matched_tracks or candidate_position in matched_candidates:
                continue
            track.add(frame_index, candidates[candidate_position], metrics)
            matched_tracks.add(track.track_id)
            matched_candidates.add(candidate_position)
        for track in active:
            if track.track_id not in matched_tracks:
                track.mark_lost(max_lost)
        for candidate_position, candidate in enumerate(candidates):
            if candidate_position in matched_candidates:
                continue
            track = Tracklet(track_id=next_track_id, start_frame=frame_index)
            next_track_id += 1
            track.add(frame_index, candidate, None)
            tracks.append(track)
    return tracks, rejected_counts, rejection_reasons


def target_track_score(
    track: Tracklet,
    num_frames: int,
    config: dict[str, Any],
    population_median_size: float | None = None,
) -> tuple[float, dict[str, float]]:
    target = _target_config(config)
    observations = [track.observations[index] for index in sorted(track.observations)]
    coverage = track.hits / max(num_frames, 1)
    completeness = float(np.mean([item.candidate.valid_count for item in observations])) / BODY_25_JOINTS
    confidence = float(np.mean([item.candidate.mean_confidence for item in observations]))
    costs = [item.match.cost for item in observations if item.match is not None]
    continuity = math.exp(-float(np.mean(costs))) if costs else 1.0
    scale_changes = [item.match.scale_change for item in observations if item.match is not None]
    scale_stability = math.exp(-float(np.mean(scale_changes))) if scale_changes else 1.0
    low_fragmentation = 1.0 / (1.0 + track.lost_events)
    components = {
        "coverage": coverage,
        "mean_valid_joints_fraction": completeness,
        "mean_confidence": confidence,
        "continuity": continuity,
        "scale_stability": scale_stability,
        "low_fragmentation": low_fragmentation,
    }
    score = (
        target.coverage_weight * coverage
        + target.valid_joints_weight * completeness
        + target.confidence_weight * confidence
        + target.continuity_weight * continuity
        + target.scale_stability_weight * scale_stability
        + target.low_fragmentation_weight * low_fragmentation
    )
    median_size = float(np.median([item.candidate.diagonal for item in observations]))
    components["median_diagonal"] = median_size
    components["neutral_score"] = score
    if target.use_size_prior and population_median_size is not None:
        # Bounded weak preference only; disabled by default.  It cannot override
        # coverage/quality by an unbounded inverse-size term.
        relative = population_median_size / max(population_median_size + median_size, 1.0)
        score += target.size_prior_weight * relative
        components["size_prior_component"] = target.size_prior_weight * relative
    else:
        components["size_prior_component"] = 0.0
    components["score"] = score
    return score, components


def track_summary(track: Tracklet, num_frames: int, config: dict[str, Any]) -> dict[str, object]:
    score, components = target_track_score(track, num_frames, config)
    return {
        "track_id": track.track_id,
        "start_frame": track.start_frame,
        "last_frame": track.last_frame,
        "hits": track.hits,
        "lost_events": track.lost_events,
        "terminated": track.terminated,
        "target_score_without_population_size_term": score,
        **components,
    }


def select_tracklet_v2(
    frames: Sequence[Sequence[np.ndarray]],
    confidence_threshold: float = 0.1,
    config: dict[str, Any] | None = None,
) -> TrackletTrackingResult:
    """Build all tracklets, then select one neutral quality/coverage target."""
    config = dict(config or {})
    tracks, rejected, rejection_reasons = build_tracklets(frames, confidence_threshold, config)
    if not tracks:
        failures = [{"frame_index": index, "reason": "no_quality_candidate"} for index in range(len(frames))]
        return TrackletTrackingResult(
            [-1] * len(frames), [None] * len(frames), failures, "tracklet_v2",
            target_track_id=-1, tracks=[], rejected_candidates_per_frame=rejected,
            config={**config, "rejection_reasons_per_frame": rejection_reasons},
        )
    median_sizes = [
        float(np.median([observation.candidate.diagonal for observation in track.observations.values()]))
        for track in tracks
    ]
    population_median_size = float(np.median(median_sizes))
    ranked = [
        (target_track_score(track, len(frames), config, population_median_size)[0], track)
        for track in tracks
    ]
    _, target = max(ranked, key=lambda item: (item[0], item[1].hits, -item[1].track_id))
    selected_indices: list[int] = []
    selected_people: list[np.ndarray | None] = []
    failures: list[dict[str, object]] = []
    for frame_index in range(len(frames)):
        observation = target.observations.get(frame_index)
        if observation is None:
            selected_indices.append(-1)
            selected_people.append(None)
            failures.append({"frame_index": frame_index, "reason": "target_track_missing"})
        else:
            selected_indices.append(observation.candidate.raw_index)
            selected_people.append(observation.candidate.person.copy())
    return TrackletTrackingResult(
        selected_indices,
        selected_people,
        failures,
        "tracklet_v2",
        target_track_id=target.track_id,
        tracks=tracks,
        rejected_candidates_per_frame=rejected,
        config={**config, "rejection_reasons_per_frame": rejection_reasons},
    )

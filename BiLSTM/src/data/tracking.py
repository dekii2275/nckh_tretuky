"""Backwards-compatible dispatcher for legacy and tracklet-v2 selection."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .child_tracking import TrackingResult, select_child_track
from .tracklet_tracking import TrackletTrackingResult, select_tracklet_v2
from .romp_guided_tracking import ROMPGuidedTrackingResult, select_romp_guided_track


def select_person_track(
    frames: Sequence[Sequence[np.ndarray]],
    confidence_threshold: float,
    data_config: dict[str, Any],
    clip_path: str | None = None,
) -> TrackingResult | TrackletTrackingResult | ROMPGuidedTrackingResult:
    tracking = data_config.get("tracking", {})
    if isinstance(tracking, str):
        tracking = {"mode": tracking}
    if not isinstance(tracking, dict):
        raise ValueError("tracking config must be a mapping or mode string")
    mode = str(tracking.get("mode", "legacy"))
    if mode == "legacy":
        # Intentionally preserve the exact pre-V2 call and defaults.
        return select_child_track(
            frames,
            confidence_threshold=confidence_threshold,
            strategy=str(data_config.get("tracking_strategy", "temporal_nearest")),
            prefer_smaller=bool(data_config.get("prefer_smaller_person", True)),
        )
    if mode == "tracklet_v2":
        return select_tracklet_v2(frames, confidence_threshold=confidence_threshold, config=tracking)
    if mode == "romp_guided":
        if clip_path is None:
            raise ValueError("romp_guided tracking requires clip_path")
        return select_romp_guided_track(
            frames,
            openpose_clip_path=clip_path,
            confidence_threshold=confidence_threshold,
            prefer_smaller=bool(data_config.get("prefer_smaller_person", True)),
            max_match_distance=float(tracking.get("max_match_distance", 0.5)),
        )
    raise ValueError(f"Unknown tracking.mode: {mode}")

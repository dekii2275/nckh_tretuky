import csv
from pathlib import Path

import numpy as np

from src.data.child_tracking import select_child_track
from src.data.openpose_parser import BODY_25_INDICES, parse_clip
from src.data.tracklet_tracking import build_tracklets, select_tracklet_v2
from src.data.tracking import select_person_track


V2 = {
    "mode": "tracklet_v2",
    "candidate_filter": {"min_valid_joints": 10, "min_torso_anchors": 4, "min_diagonal_px": 50},
    "gating": {
        "max_normalized_centroid_distance": 0.5,
        "max_scale_ratio": 1.8,
        "max_pose_distance": 0.35,
        "min_shared_joints": 6,
    },
    "tracks": {"max_lost_frames": 3},
    "target_selection": {"use_size_prior": False},
}


def full_person(x, y=100.0, scale=4.0, confidence=0.9, signature=0.0):
    person = np.zeros((25, 3), dtype=np.float32)
    columns = np.asarray([(index % 5) - 2 for index in range(25)], dtype=np.float32)
    rows = np.asarray([index // 5 for index in range(25)], dtype=np.float32)
    person[:, 0] = x + columns * scale
    person[:, 1] = y + rows * scale * 5
    person[:, 2] = confidence
    person[BODY_25_INDICES["RWrist"], 1] += signature
    return person


def fragment(x=10.0, y=10.0):
    person = np.zeros((25, 3), dtype=np.float32)
    person[0] = [x, y, 0.95]
    person[1] = [x + 5, y + 5, 0.95]
    return person


def test_case_a_full_skeleton_beats_fragment():
    frames = [[fragment(), full_person(200 + time)] for time in range(5)]
    result = select_tracklet_v2(frames, config=V2)
    assert result.selected_indices == [1] * 5
    assert result.rejected_candidates_per_frame == [1] * 5


def test_case_b_same_person_smooth_motion():
    frames = [[full_person(100 + 3 * time)] for time in range(8)]
    result = select_tracklet_v2(frames, config=V2)
    assert result.selected_indices == [0] * 8
    assert not result.failures
    assert len([track for track in result.tracks if track.hits]) == 1


def test_case_c_crossing_tracks_keep_pose_identity():
    frames = []
    for time, (a_x, b_x) in enumerate(zip((100, 140, 180, 220), (260, 220, 180, 140))):
        a = full_person(a_x, confidence=0.91, signature=45)
        b = full_person(b_x, confidence=0.71, signature=-35)
        frames.append([a, b] if time % 2 == 0 else [b, a])
    tracks, _, _ = build_tracklets(frames, 0.1, V2)
    long_tracks = [track for track in tracks if track.hits == 4]
    assert len(long_tracks) == 2
    confidence_sequences = [
        [round(observation.candidate.mean_confidence, 2) for observation in track.observations.values()]
        for track in long_tracks
    ]
    assert sorted(confidence_sequences) == [[0.71] * 4, [0.91] * 4]


def test_case_d_raw_candidate_order_does_not_define_track():
    a0, b0 = full_person(100, confidence=0.9), full_person(300, confidence=0.7)
    a1, b1 = full_person(103, confidence=0.9), full_person(297, confidence=0.7)
    tracks, _, _ = build_tracklets([[a0, b0], [b1, a1]], 0.1, V2)
    long_tracks = sorted((track for track in tracks if track.hits == 2), key=lambda track: track.track_id)
    assert len(long_tracks) == 2
    assert [long_tracks[0].observations[t].candidate.raw_index for t in (0, 1)] == [0, 1]
    assert [long_tracks[1].observations[t].candidate.raw_index for t in (0, 1)] == [1, 0]


def test_case_e_target_lost_then_reacquired_without_substitution():
    frames = [[full_person(100)], [full_person(103)], [], [], [full_person(112)], [full_person(115)]]
    result = select_tracklet_v2(frames, config=V2)
    assert result.selected_indices == [0, 0, -1, -1, 0, 0]
    assert [failure["frame_index"] for failure in result.failures] == [2, 3]
    assert result.target_track_id == 0


def test_case_f_impossible_jump_is_not_force_matched():
    frames = [[full_person(100)], [full_person(1000)]]
    tracks, _, _ = build_tracklets(frames, 0.1, V2)
    assert sorted(track.hits for track in tracks) == [1, 1]


def test_case_g_one_person_clip():
    frames = [[full_person(200 + time)] for time in range(4)]
    result = select_person_track(frames, 0.1, {"tracking": V2})
    assert result.strategy == "tracklet_v2"
    assert result.selected_indices == [0, 0, 0, 0]


def test_legacy_dispatch_is_bitwise_identical_on_real_subset():
    root = Path(__file__).resolve().parents[1]
    manifest = root / "artifacts/paper_repro/manifest.csv"
    if not manifest.exists():
        return
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    chosen = [rows[0], rows[len(rows) // 2], rows[-1]]
    data_config = {"tracking_strategy": "temporal_nearest", "prefer_smaller_person": True}
    for row in chosen:
        frames, _ = parse_clip(row["path"], strict=True)
        direct = select_child_track(frames, 0.1, "temporal_nearest", True)
        dispatched = select_person_track(frames, 0.1, data_config)
        assert dispatched.selected_indices == direct.selected_indices
        assert dispatched.failures == direct.failures
        for actual, expected in zip(dispatched.selected_people, direct.selected_people, strict=True):
            if expected is None:
                assert actual is None
            else:
                assert np.array_equal(actual, expected)

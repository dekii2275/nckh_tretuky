"""Regression cases for the multi-person review association and source IDs."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from romp_review_data import MATCH_JOINTS, frame_id, match_selected_person, read_romp_frames


def pair(offset=0):
    romp = np.zeros((24, 3), dtype=float)
    op = np.zeros((25, 3), dtype=float)
    for i, (smpl, body) in enumerate(MATCH_JOINTS):
        xy = [20 + (i % 2) * 60 + offset, 30 + (i // 2) * 30]
        romp[smpl] = [*xy, 100]
        op[body] = [*xy, 0.9]
    return op, romp.tolist()


def test_selected_index_is_matched_by_joints_not_array_order():
    op, correct = pair()
    _, other = pair(200)
    match = match_selected_person(op, [other, correct])
    assert match == {"index": 1, "status": "matched", "score": 0.0}
    assert match_selected_person(op, [correct, other])["index"] == 0


def test_missing_or_ambiguous_person_is_not_forced_to_match():
    op, correct = pair()
    _, far = pair(300)
    assert match_selected_person(op, [far])["status"] == "unmatched"
    assert match_selected_person(op, [correct, correct])["status"] == "ambiguous"
    op[:, 2] = 0
    assert match_selected_person(op, [correct])["index"] == -1


def test_partial_invalid_pose_keeps_person_slot_and_valid_joints(tmp_path):
    _, correct = pair()
    first = np.asarray(correct); first[4] = np.nan
    np.savez(tmp_path / "00000037.npz", coordinates=np.asarray([first, correct]))
    frames, errors = read_romp_frames(tmp_path)
    assert not errors
    assert len(frames[37]) == 2
    assert frames[37][0][4] is None
    assert frames[37][0][16] == correct[16]
    assert frames[37][1] == correct


def test_unsupported_shape_has_reported_error_and_keeps_frame(tmp_path):
    np.savez(tmp_path / "00000038.npz", coordinates=np.zeros((1, 25, 3)))
    frames, errors = read_romp_frames(tmp_path)
    assert frames == {38: []}
    assert errors[0]["frame_id"] == 38
    assert "Unsupported" in errors[0]["reason"]


def test_romp71_uses_documented_smpl24_core(tmp_path):
    coords = np.arange(213, dtype=float).reshape(1, 71, 3)
    np.savez(tmp_path / "00000000.npz", coordinates=coords)
    frames, errors = read_romp_frames(tmp_path)
    assert not errors
    assert frames[0] == coords[:, :24].tolist()
    assert frame_id(Path("as_20613_D1_002_i_000000000037_keypoints.json")) == 37


def test_real_screenshot_frame_has_independent_romp_index():
    root = Path(__file__).resolve().parents[2] / "data_MMASD"
    clip = "as_20613_D1_002_i"
    npz = root / "3D_skeleton/ROMP_3D_Coordinates/arm_swing_as" / clip / "00000025.npz"
    source = root / "2D_skeleton/2D_openpose/output/arm_swing_as" / clip / f"{clip}_000000000025_keypoints.json"
    if not npz.is_file() or not source.is_file():
        import pytest
        pytest.skip("Local MMASD source files are unavailable")
    from src.data.openpose_parser import parse_frame
    with np.load(npz) as data:
        people = data["coordinates"].tolist()
    match = match_selected_person(parse_frame(source)[3], people)
    assert match["status"] == "matched"
    assert match["index"] == 4  # The old viewer incorrectly highlighted ROMP 3.

import numpy as np

from src.data.child_tracking import select_child_track
from src.data.openpose_parser import BODY_25_INDICES


def person(x, scale, confidence=0.9):
    value = np.zeros((25, 3), dtype=np.float32)
    value[:, 2] = confidence
    value[:, 0] = x
    value[:, 1] = np.arange(25, dtype=np.float32) * scale
    value[BODY_25_INDICES["MidHip"], :2] = [x, 10 * scale]
    return value


def test_tracking_is_deterministic_and_not_people_zero():
    frames = [
        [person(100, 10), person(10, 4)],
        [person(102, 10), person(11, 4)],
    ]
    first = select_child_track(frames)
    second = select_child_track(frames)
    assert first.selected_indices == second.selected_indices
    assert first.selected_indices == [1, 1]
    assert first.debug_dict("clip") ["clip_id"] == "clip"


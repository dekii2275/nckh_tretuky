import numpy as np

from src.data.openpose_parser import BODY_25_INDICES
from src.data.preprocessing import preprocess_clip


def make_person(frames=3):
    people = []
    for frame in range(frames):
        person = np.zeros((25, 3), dtype=np.float32)
        person[:, 2] = 0.9
        person[:, :2] = [10 + frame, 20 + frame]
        person[BODY_25_INDICES["MidHip"], :2] = [5 + frame, 10 + frame]
        person[BODY_25_INDICES["LShoulder"], :2] = [3 + frame, 10 + frame]
        person[BODY_25_INDICES["RShoulder"], :2] = [7 + frame, 10 + frame]
        people.append(person)
    return people


def test_midhip_zero_and_shoulder_width_one():
    result = preprocess_clip(make_person())
    midhip = BODY_25_INDICES["MidHip"]
    left = BODY_25_INDICES["LShoulder"]
    right = BODY_25_INDICES["RShoulder"]
    assert np.allclose(result.coordinates[:, midhip], 0.0, atol=1e-6)
    widths = np.linalg.norm(result.coordinates[:, left] - result.coordinates[:, right], axis=1)
    assert np.allclose(widths, 1.0, atol=1e-6)


def test_invalid_shoulders_use_logged_fallback():
    people = make_person(2)
    people[1][BODY_25_INDICES["LShoulder"]] = [0, 0, 0]
    people[1][BODY_25_INDICES["RShoulder"]] = [0, 0, 0]
    result = preprocess_clip(people)
    assert result.metadata["scale_fallback_frames"] >= 1
    assert np.isfinite(result.coordinates).all()


def test_near_zero_shoulder_width_uses_torso_fallback():
    people = make_person(2)
    left = BODY_25_INDICES["LShoulder"]
    right = BODY_25_INDICES["RShoulder"]
    people[1][left, :2] = [5.0, 10.0]
    people[1][right, :2] = [5.001, 10.0]
    result = preprocess_clip(people, min_shoulder_width=5.0)
    assert result.metadata["scale_fallback_frames"] >= 1
    assert np.isfinite(result.coordinates).all()
    assert float(np.abs(result.coordinates).max()) < 100.0


def test_interpolation_and_mask_strategy():
    people = make_person(3)
    missing_joint = 4
    people[1][missing_joint] = [0, 0, 0]
    interpolated = preprocess_clip(people, missing_joint_strategy="interpolate")
    assert interpolated.valid_mask[1, missing_joint] is False or not interpolated.valid_mask[1, missing_joint]
    masked = preprocess_clip(people, missing_joint_strategy="mask")
    assert np.allclose(masked.coordinates[1, missing_joint], 0.0)

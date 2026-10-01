import json

import numpy as np

from src.data.openpose_parser import BODY_25_JOINTS, parse_frame


def test_openpose_body25_parser(tmp_path):
    values = []
    for index in range(BODY_25_JOINTS):
        values.extend([float(index), float(index + 1), 0.9])
    path = tmp_path / "frame_keypoints.json"
    path.write_text(json.dumps({"version": 1.3, "people": [{"pose_keypoints_2d": values}]}))
    people = parse_frame(path)
    assert len(people) == 1
    assert people[0].shape == (25, 3)
    assert np.allclose(people[0][3], [3.0, 4.0, 0.9])


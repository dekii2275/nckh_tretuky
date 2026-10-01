import torch

from src.data.collate import skeleton_collate
from src.data.splits import split_records, validate_no_leakage


def test_variable_length_batching_and_padding_mask():
    batch = [
        {"skeleton": torch.ones(3, 25, 2), "label": 0, "clip_id": "a", "path": "a"},
        {"skeleton": torch.ones(5, 25, 2), "label": 1, "clip_id": "b", "path": "b"},
    ]
    output = skeleton_collate(batch)
    assert output["skeleton"].shape == (2, 5, 25, 2)
    assert output["lengths"].tolist() == [3, 5]
    assert output["padding_mask"].tolist() == [[False, False, False, True, True], [False] * 5]


def test_no_split_leakage_for_session_groups():
    rows = [
        {"clip_id": "a", "path": "a", "subject_id": "s1", "session_id": "D1", "session_group": "s1::D1", "activity": "a", "activity_id": "0", "num_frames": "1"},
        {"clip_id": "b", "path": "b", "subject_id": "s1", "session_id": "D1", "session_group": "s1::D1", "activity": "b", "activity_id": "1", "num_frames": "1"},
        {"clip_id": "c", "path": "c", "subject_id": "s2", "session_id": "D1", "session_group": "s2::D1", "activity": "a", "activity_id": "0", "num_frames": "1"},
        {"clip_id": "d", "path": "d", "subject_id": "s3", "session_id": "D8", "session_group": "s3::D8", "activity": "b", "activity_id": "1", "num_frames": "1"},
        {"clip_id": "e", "path": "e", "subject_id": "s4", "session_id": "D8", "session_group": "s4::D8", "activity": "a", "activity_id": "0", "num_frames": "1"},
    ]
    splits = split_records(rows, seed=3, val_fraction=0.2, test_fraction=0.2)
    validate_no_leakage(splits, "session_holdout")
    for split in splits.values():
        assert len({row["clip_id"] for row in split}) == len(split)


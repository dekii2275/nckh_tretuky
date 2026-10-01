from types import SimpleNamespace

import numpy as np
import pytest

from src.data.dataset import SkeletonClipDataset
from src.data.sequence import SequenceBuilder, selected_source_indices
from src.data.splits import validate_no_leakage
from src.sequence_study import aggregate_clip_logits


def row(clip_id="clip", frames=120, subject="s", session="D1"):
    return {"clip_id": clip_id, "subject_id": subject, "session_id": session,
            "session_group": f"{subject}::{session}", "activity": "a", "activity_id": 0,
            "num_frames": frames, "path": f"/{clip_id}"}


def test_n60_contiguous_preserves_all_frames():
    builder = SequenceBuilder({"mode": "single_window", "length": 60})
    source = np.arange(60 * 50, dtype=np.float32).reshape(60, 25, 2)
    meta = builder.describe(row(frames=60))[0]
    assert np.array_equal(builder.extract(source, meta), source)


@pytest.mark.parametrize("mode, starts", [
    ("window_60_stride_60", [0, 60]),
    ("window_60_stride_30", [0, 30, 60]),
])
def test_expected_windows_for_n120(mode, starts):
    windows = SequenceBuilder({"mode": mode, "length": 60}).describe(row())
    assert [item["start_frame"] for item in windows] == starts
    assert [item["end_frame"] for item in windows] == [start + 59 for start in starts]


def test_center_crop_boundaries():
    meta = SequenceBuilder({"mode": "single_window", "length": 60}).describe(row(frames=121))[0]
    assert meta["start_frame"] == 30 and meta["end_frame"] == 89


def test_short_clip_interpolation_policy_and_metadata():
    builder = SequenceBuilder({"mode": "window_60_stride_30", "length": 60,
                               "short_clip_policy": "interpolate"})
    source = np.arange(20 * 50, dtype=np.float32).reshape(20, 25, 2)
    meta = builder.describe(row(frames=20))[0]
    result = builder.extract(source, meta)
    assert result.shape == (60, 25, 2)
    assert meta == {"source_clip_id": "clip", "sequence_id": "clip__start0000_end0019",
                    "start_frame": 0, "end_frame": 19, "original_num_frames": 20,
                    "window_index": 0, "sequence_mode": "window_60_stride_30",
                    "is_interpolated_short_clip": True}


def test_validation_mean_logits_aggregation():
    metadata = [{"source_clip_id": "x", "clip_id": "x", "subject_id": "s",
                 "session_id": "D1", "activity": "a"}] * 3
    truth, pred, rows = aggregate_clip_logits([[2, 0], [2, 0], [0, 9]], metadata, [1, 1, 1])
    assert truth == [1] and pred == [1]
    assert rows[0]["num_sequences"] == 3
    assert np.isclose(rows[0]["mean_logit_1"], 3.0)


def test_split_before_expansion_has_no_source_overlap():
    splits = {"train": [row("a", subject="s1")], "val": [row("b", subject="s2")]}
    validate_no_leakage(splits, "session_holdout")
    builder = SequenceBuilder({"mode": "window_60_stride_30"})
    train = {item["source_clip_id"] for item in builder.expand(splits["train"])}
    val = {item["source_clip_id"] for item in builder.expand(splits["val"])}
    assert not train & val


def test_tracker_preprocessing_output_exists_unchanged_before_sampling(monkeypatch):
    tracked = [object()] * 120
    normalized = np.arange(120 * 50, dtype=np.float32).reshape(120, 25, 2)
    calls = []
    monkeypatch.setattr("src.data.dataset.parse_clip", lambda *args, **kwargs: ([object()] * 120, {}))
    def fake_track(frames, threshold, config, **kwargs):
        calls.append("track")
        return SimpleNamespace(selected_people=tracked, debug_dict=lambda clip: {})
    def fake_preprocess(people, **kwargs):
        assert people is tracked
        calls.append("preprocess")
        return SimpleNamespace(coordinates=normalized, metadata={}, valid_mask=np.ones((120, 25), bool))
    monkeypatch.setattr("src.data.dataset.select_person_track", fake_track)
    monkeypatch.setattr("src.data.dataset.preprocess_clip", fake_preprocess)
    dataset = SkeletonClipDataset([row()], sequence_config={"mode": "full_clip_resample", "length": 60})
    item = dataset[0]
    assert calls == ["track", "preprocess"]
    assert np.array_equal(normalized, np.arange(120 * 50, dtype=np.float32).reshape(120, 25, 2))
    assert np.array_equal(item["skeleton"].numpy(), normalized[selected_source_indices(120)])


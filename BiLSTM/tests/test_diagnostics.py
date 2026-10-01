import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_tracking_labels import label_rates, performance
from analyze_child_size import comparison, size
from audit_preprocessing_usage import missing_runs, summarize
from diagnostics_common import read_manifest


def test_unreviewed_clips_never_get_human_group_metrics():
    rows = [{"clip_id": "x", "review_label": ""}]
    predictions = {"x": {"true_id": "0", "predicted_id": "0"}}
    rates = label_rates(rows)
    assert rates["reviewed"] == 0 and rates["unreviewed"] == 1
    assert rates["correct_child_rate"] is None
    for label in ("A", "B", "C", "D", "B+C"):
        result = performance(rows, predictions, label)
        assert result["s1_validation_clips"] == 0
        assert result["accuracy"] is None and result["macro_f1"] is None


def test_human_rates_and_validation_only_scores():
    rows = [{"clip_id": str(i), "review_label": label} for i, label in enumerate(["A", "B", "C", "D", ""])]
    rates = label_rates(rows)
    assert rates["correct_child_rate"] == rates["wrong_person_rate"] == rates["identity_switch_rate"] == rates["uncertain_rate"] == 0.25
    result = performance(rows, {"1": {"true_id": "0", "predicted_id": "1"}}, "B+C")
    assert result["reviewed_clips_all_splits"] == 2
    assert result["s1_validation_clips"] == 1 and result["accuracy"] == 0


def test_human_csv_roundtrip_accepts_bom_quotes_and_newlines(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text('\ufeffclip_id,review_label,notes\r\nx,A,"comma, quote ""yes""\nnew line"\r\n')
    assert read_manifest(path) == [{"clip_id": "x", "review_label": "A", "notes": 'comma, quote "yes"\nnew line'}]


def test_gaps_distinguish_interpolation_edges_and_never_observed():
    valid = np.array([[0, 0], [1, 0], [0, 0], [1, 0], [0, 0]], dtype=bool)
    assert sorted(missing_runs(valid)) == sorted([("leading_edge", 1), ("bounded_gap", 1), ("trailing_edge", 1), ("never_observed_joint", 5)])


def test_preprocessing_frame_weighting_and_missingness_count_not_overwritten():
    def row(t, invalid):
        r = dict.fromkeys(["invalid_midhip_frames", "missing_left_shoulder_frames", "missing_right_shoulder_frames", "invalid_shoulder_scale_frames", "torso_fallback_frames", "frames_gt25pct_joints_missing", "missing_joints", "interpolated_joints", "bounded_interpolated_joints", "edge_filled_joints", "never_observed_joint_frames", "severe_missingness"], 0)
        r.update(num_frames=t, invalid_midhip_frames=invalid, frames_gt25pct_joints_missing=invalid)
        return r
    result = summarize([row(10, 10), row(90, 0)])
    assert result["invalid_midhip_percent"] == 10  # not the unweighted clip mean of 50%
    assert result["frames_gt25pct_joints_missing"] == 10
    assert result["frames_gt25pct_joints_missing_percent"] == 10


def test_size_ignores_invalid_joints_and_handles_ambiguity():
    person = np.zeros((25, 3))
    person[:2] = [[10, 10, 1], [20, 30, 1]]
    person[2] = [1000, 1000, 0]
    assert size(person) == (200, 20)
    assert comparison(None) == comparison(1.0) == "ambiguous"
    assert comparison(0.9) == "child_smaller"
    assert comparison(1.1) == "child_larger"

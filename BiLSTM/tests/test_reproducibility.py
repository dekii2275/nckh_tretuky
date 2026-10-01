from copy import deepcopy
from pathlib import Path

import pytest

from src.reproducibility import capture_provenance, verify_checkpoint_inputs
from src.utils import load_config
from src.data.splits import validate_no_leakage


def fixture_config(tmp_path):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/paper_repro.yaml")
    config["outputs"] = {"artifacts_dir": str(tmp_path), "split_dir": str(tmp_path / "splits")}
    (tmp_path / "splits").mkdir()
    for name in ("manifest.csv", "splits/train.csv", "splits/val.csv"):
        (tmp_path / name).write_text("clip_id,activity\na,arm_swing\n")
    return config


def test_evaluation_rejects_changed_split_even_when_config_is_identical(tmp_path):
    config = fixture_config(tmp_path)
    saved = capture_provenance(config)
    checkpoint = {"config": config, "provenance": saved}
    assert verify_checkpoint_inputs(checkpoint, config, saved)["status"] == "recorded_manifests_verified"
    (tmp_path / "splits/val.csv").write_text("clip_id,activity\nb,arm_swing\n")
    with pytest.raises(ValueError, match="Manifest/split"):
        verify_checkpoint_inputs(checkpoint, config, capture_provenance(config))


def test_evaluation_rejects_different_sampling_or_preprocessing(tmp_path):
    config = fixture_config(tmp_path)
    checkpoint = {"config": deepcopy(config), "provenance": capture_provenance(config)}
    config["sequence"]["length"] = 30
    with pytest.raises(ValueError, match="sequence"):
        verify_checkpoint_inputs(checkpoint, config, capture_provenance(config))
    config["sequence"]["length"] = 60
    config["data"]["preprocessing"]["confidence_threshold"] = 0.5
    with pytest.raises(ValueError, match="data"):
        verify_checkpoint_inputs(checkpoint, config, capture_provenance(config))


def test_evaluation_rejects_different_tracker_mode(tmp_path):
    config = fixture_config(tmp_path)
    config["tracking"] = {"mode": "legacy"}
    checkpoint = {"config": deepcopy(config), "provenance": capture_provenance(config)}
    config["tracking"] = {"mode": "tracklet_v2"}
    with pytest.raises(ValueError, match="tracking"):
        verify_checkpoint_inputs(checkpoint, config, capture_provenance(config))


def test_old_checkpoint_does_not_get_retroactive_split_verification(tmp_path):
    config = fixture_config(tmp_path)
    result = verify_checkpoint_inputs({"config": config}, config, capture_provenance(config))
    assert result["status"] == "legacy_checkpoint_manifest_unverified"


def test_source_changes_are_reported_separately_from_manifest_integrity(tmp_path):
    config = fixture_config(tmp_path)
    saved = capture_provenance(config)
    changed = deepcopy(saved)
    changed["source_sha256"]["engine.py"] = "different"
    result = verify_checkpoint_inputs({"config": config, "provenance": saved}, config, changed)
    assert result["source_matches"] is False


def test_group_validation_rejects_subject_leakage_and_duplicate_rows():
    row = {"clip_id": "a", "path": "a", "subject_id": "s1", "session_id": "D1", "session_group": "s1::D1"}
    second = {**row, "clip_id": "b", "path": "b", "session_id": "D8", "session_group": "s1::D8"}
    with pytest.raises(RuntimeError, match="subject_holdout"):
        validate_no_leakage({"train": [row], "val": [second]}, "subject_holdout")
    with pytest.raises(RuntimeError, match="Duplicate"):
        validate_no_leakage({"train": [row, row]}, "session_holdout")
    with pytest.raises(RuntimeError, match="Missing group identity"):
        validate_no_leakage({"train": [{**row, "subject_id": "unknown"}]}, "session_holdout")

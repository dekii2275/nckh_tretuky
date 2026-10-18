"""Protect raw observations and snapshot integrity without private datasets."""

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_pipeline_lock as audit  # noqa: E402
from src.data.openpose_parser import OpenPoseFormatError  # noqa: E402
from src.reproducibility import sha256_file  # noqa: E402
from src.utils import load_config  # noqa: E402


def clip_fixture(tmp_path):
    config = load_config(audit.ROOT / "configs/skeleton_audit_v0_2.yaml")
    config["data"]["root"] = str(tmp_path / "data")
    directory = tmp_path / "data" / config["data"]["skeleton_dir"] / "arm_swing_as" / "sample"
    directory.mkdir(parents=True)
    person = np.zeros((25, 3), dtype=np.float32)
    person[:, 0] = 100 + (np.arange(25) % 5) * 15
    person[:, 1] = 100 + (np.arange(25) // 5) * 20
    person[:, 2] = 0.9
    person[4, 2] = 0.05
    for index, people in enumerate(([person], [], [person])):
        payload = {"people": [{"pose_keypoints_2d": item.reshape(-1).tolist()} for item in people]}
        path = directory / f"sample_{index:012d}_keypoints.json"
        path.write_text(json.dumps(payload))
    record = {
        "path": "/old-machine/unavailable",
        "activity": "arm_swing_as",
        "clip_id": "sample",
        "num_frames": 3,
    }
    return config, record, directory, person


def test_raw_observation_preserves_confidence_and_missing_targets(tmp_path):
    config, record, directory, person = clip_fixture(tmp_path)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    _, paths, selected, raw, valid = audit.observe_raw(record, config)
    assert selected.selected_indices == [0, -1, 0]
    assert len(paths) == 3 and raw.shape == (3, 25, 3)
    assert np.array_equal(raw[0], person)
    assert raw[0, 4, 2] == person[4, 2] and not valid[0, 4]
    assert np.isnan(raw[1]).all() and not valid[1].any()
    assert before == {path.name: path.read_bytes() for path in directory.iterdir()}


def test_raw_observation_rejects_frame_count_drift_and_malformed_person(tmp_path):
    config, record, directory, _ = clip_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="Manifest/frame mismatch"):
        audit.observe_raw({**record, "num_frames": 4}, config)
    first = sorted(directory.iterdir())[0]
    first.write_text(json.dumps({"people": [{"pose_keypoints_2d": [1] * 72}]}))
    with pytest.raises(OpenPoseFormatError, match="expected 75"):
        audit.observe_raw(record, config)


def test_raw_observation_refuses_sampling_or_augmentation(tmp_path):
    config, record, _, _ = clip_fixture(tmp_path)
    config["sequence"]["mode"] = "fixed_60"
    with pytest.raises(RuntimeError, match="requires full_clip"):
        audit.observe_raw(record, config)
    config["sequence"]["mode"] = "full_clip"
    config["augmentation"]["enabled"] = True
    with pytest.raises(RuntimeError, match="forbids augmentation"):
        audit.observe_raw(record, config)


def snapshot_fixture(tmp_path):
    source = tmp_path / "BiLSTM/src/parser.py"
    source.parent.mkdir(parents=True)
    source.write_text("VERSION = 1\n")
    config = {"seed": 42}
    lock = {
        "schema_version": "skeleton_pipeline_lock_v0_2",
        "resolved_config": config,
        "files_sha256": {"BiLSTM/src/parser.py": sha256_file(source)},
        "source_inventory": ["BiLSTM/src/parser.py"],
        "smoke_frame_inventory": [],
        "environment": audit.environment_versions(),
    }
    output = tmp_path / "snapshot"
    audit.write_snapshot(output, {"report.json": {"ok": True}}, lock)
    return output, config, source


def test_snapshot_roundtrip_and_no_overwrite(tmp_path):
    output, config, _ = snapshot_fixture(tmp_path)
    assert audit.verify_snapshot(output, config, repo=tmp_path) == 1
    with pytest.raises(RuntimeError, match="Output exists"):
        audit.write_snapshot(output, {}, {})
    assert audit.read_json(output / "report.json") == {"ok": True}


@pytest.mark.parametrize("change", ["source", "new_source", "config", "output", "runtime"])
def test_snapshot_rejects_drift(tmp_path, change):
    output, config, source = snapshot_fixture(tmp_path)
    if change == "source":
        source.write_text("VERSION = 2\n")
    elif change == "new_source":
        (source.parent / "extra.py").write_text("\n")
    elif change == "config":
        config = {"seed": 43}
    elif change == "output":
        (output / "report.json").write_text('{"ok": false}')
    else:
        lock = audit.read_json(output / "pipeline_lock.json")
        lock["environment"]["numpy"] = "different"
        audit.write_json(output / "pipeline_lock.json", lock)
    with pytest.raises(RuntimeError):
        audit.verify_snapshot(output, config, repo=tmp_path)


def test_snapshot_checks_frame_inventory_even_when_known_file_hashes_match(tmp_path):
    output, config, _ = snapshot_fixture(tmp_path)
    directory = tmp_path / "clip"
    directory.mkdir()
    first = directory / "clip_000000000000_keypoints.json"
    first.write_text('{"people": []}')
    lock = audit.read_json(output / "pipeline_lock.json")
    lock["files_sha256"]["clip/" + first.name] = sha256_file(first)
    lock["smoke_frame_inventory"] = [{"path": "clip", "frames": ["clip/" + first.name]}]
    audit.write_json(output / "pipeline_lock.json", lock)
    assert audit.verify_snapshot(output, config, repo=tmp_path) == 2
    (directory / "clip_000000000001_keypoints.json").write_text('{"people": []}')
    with pytest.raises(RuntimeError, match="Smoke frame inventory changed"):
        audit.verify_snapshot(output, config, repo=tmp_path)


def test_snapshot_never_reads_paths_outside_repository(tmp_path):
    output, config, _ = snapshot_fixture(tmp_path)
    lock = audit.read_json(output / "pipeline_lock.json")
    lock["files_sha256"] = {"../outside": "0" * 64}
    audit.write_json(output / "pipeline_lock.json", lock)
    with pytest.raises(RuntimeError, match="outside repository"):
        audit.verify_snapshot(output, config, repo=tmp_path)


def test_failed_json_serialization_does_not_publish_snapshot(tmp_path):
    output = tmp_path / "failed_snapshot"
    with pytest.raises(ValueError):
        audit.write_snapshot(output, {"bad.json": {"value": float("nan")}}, {})
    assert not output.exists()


def test_config_hashes_include_base_files_and_reject_cycles(tmp_path):
    parent = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    parent.write_text("seed: 42\n")
    child.write_text("base_config: base.yaml\nsequence: {mode: full_clip}\n")
    recorded = {}

    def remember(path):
        recorded[path.name] = sha256_file(path)

    audit.remember_config(child, remember)
    assert set(recorded) == {"base.yaml", "child.yaml"}
    original = deepcopy(recorded)
    parent.write_text("seed: 43\n")
    audit.remember_config(child, remember)
    assert recorded["base.yaml"] != original["base.yaml"]
    parent.write_text("base_config: child.yaml\n")
    with pytest.raises(RuntimeError, match="Circular config dependency"):
        audit.remember_config(child, remember)

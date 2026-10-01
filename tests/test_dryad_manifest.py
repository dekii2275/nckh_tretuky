"""Round 3 canonical manifest invariants."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "dryad_msg3d"


def read_manifest():
    with (ARTIFACT_ROOT / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_manifest_counts_and_required_columns():
    rows = read_manifest()
    required = {
        "sample_id", "class_name", "class_id", "subject_id", "source",
        "is_original", "is_augmented", "augmentation_id", "original_sample_id",
        "source_path", "raw_sha256", "numeric_hash", "raw_frame_count",
        "valid_frame_count", "target_frame_count", "raw_joint_count",
        "raw_coordinate_count", "coord_order", "joint_mapping_version",
        "preprocessing_version", "m_policy", "had_sentinel", "sentinel_count",
        "repair_method", "duplicate_group_id", "duplicate_component_id",
    }
    assert required.issubset(rows[0])
    assert len(rows) == 800
    assert {row["class_name"] for row in rows} == {"ASD", "TD"}
    assert sum(row["class_name"] == "ASD" for row in rows) == 400
    assert sum(row["class_name"] == "TD" for row in rows) == 400
    assert sum(row["is_original"] == "1" for row in rows) == 100
    assert sum(row["is_augmented"] == "1" for row in rows) == 700
    assert len({row["subject_id"] for row in rows}) == 100
    assert len({row["sample_id"] for row in rows}) == 800
    assert all(row["numeric_hash"] for row in rows)
    assert all(row["coord_order"] == "XYZ" for row in rows)
    assert all(row["target_frame_count"] == "60" for row in rows)


def test_augmentation_provenance_is_subject_and_class_consistent():
    rows = read_manifest()
    by_sample = {row["sample_id"]: row for row in rows}
    for row in rows:
        if row["is_augmented"] != "1":
            assert row["original_sample_id"] == row["sample_id"]
            continue
        original = by_sample[row["original_sample_id"]]
        assert original["is_original"] == "1"
        assert original["subject_id"] == row["subject_id"]
        assert original["class_name"] == row["class_name"]
        assert row["original_sample_id"].endswith("_original")


def test_manifest_sentinel_and_duplicate_counts():
    rows = read_manifest()
    assert sum(int(row["sentinel_count"]) for row in rows) == 101
    assert sum(row["had_sentinel"] == "1" for row in rows) == 3
    assert {row["subject_id"] for row in rows if row["had_sentinel"] == "1"} == {
        "ASD_004", "ASD_008", "ASD_022"
    }
    assert sum(bool(row["duplicate_group_id"]) for row in rows) == 117
    assert len({row["duplicate_component_id"] for row in rows}) == 88

"""Random-fold and leakage assertions for both Round 3 protocols."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "dryad_msg3d"


def folds(protocol):
    root = ARTIFACT_ROOT / "folds" / protocol
    return [json.loads((root / f"fold_{index:02d}.json").read_text(encoding="utf-8")) for index in range(10)]


def test_both_protocols_have_balanced_random_10_folds():
    for protocol in ("random_reproduction_like", "random_leakage_safe"):
        values = folds(protocol)
        assert len(values) == 10
        for fold in values:
            assert fold["train_subject_count"] == 90
            assert fold["test_subject_count"] == 10
            assert fold["train_class_counts"] == {"ASD": 360, "TD": 360}
            assert fold["test_class_counts"] == {"ASD": 40, "TD": 40}
            assert not set(fold["train_subject_ids"]) & set(fold["test_subject_ids"])
            assert not set(fold["train_sample_ids"]) & set(fold["test_sample_ids"])


def test_leakage_safe_has_zero_component_and_hash_leakage():
    for fold in folds("random_leakage_safe"):
        assert fold["duplicate_leakage_count"] == 0
        assert fold["train_test_component_intersection"] == []
        assert fold["train_test_numeric_hash_intersection"] == []


def test_reproduction_like_reports_duplicate_leakage_without_failing():
    values = folds("random_reproduction_like")
    leakage_counts = [fold["duplicate_leakage_count"] for fold in values]
    assert any(count > 0 for count in leakage_counts)
    assert all(count >= 0 for count in leakage_counts)

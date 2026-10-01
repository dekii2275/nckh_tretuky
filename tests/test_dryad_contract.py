"""Contract-level checks; no training and no raw-data mutation."""

import csv
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from dryad_contract import (  # noqa: E402
    CONTRACT_VERSION,
    SENTINEL_VALUE,
    sentinel_mask,
    stack_zero_person,
)


def test_person_and_temporal_contract_are_explicit():
    source = np.arange(3 * 34 * 25, dtype=np.float32).reshape(3, 34, 25)
    original = source.copy()
    tensor = stack_zero_person(source)
    assert tensor.shape == (3, 60, 25, 2)
    np.testing.assert_array_equal(tensor[:, :34, :, 0], source)
    assert np.all(tensor[:, 34:, :, 0] == 0)
    assert np.all(tensor[:, :, :, 1] == 0)
    np.testing.assert_array_equal(source, original)


def test_temporal_policy_rejects_unexpected_truncation():
    with pytest.raises(ValueError, match="exceeds"):
        stack_zero_person(np.zeros((3, 61, 25), dtype=np.float32))


def test_sentinel_detection_is_exact_and_non_destructive():
    source = np.array([[[99.99999, 100.0, 100.00001]]], dtype=np.float32)
    original = source.copy()
    mask = sentinel_mask(source)
    assert mask.tolist() == [[[False, True, False]]]
    assert SENTINEL_VALUE == 100.0
    np.testing.assert_array_equal(source, original)


def test_duplicate_artifacts_match_scan_summary():
    artifact_dir = ROOT / "artifacts" / "dryad_msg3d"
    with (artifact_dir / "duplicate_groups.csv").open(newline="", encoding="utf-8") as handle:
        duplicate_rows = list(csv.DictReader(handle))
    with (artifact_dir / "duplicate_subject_components.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        component_rows = list(csv.DictReader(handle))
    assert len({row["duplicate_group_id"] for row in duplicate_rows}) == 44
    assert len(duplicate_rows) == 117
    assert len(component_rows) == 100
    assert len({row["component_id"] for row in component_rows}) == 88
    assert max(
        len(row["connected_subjects"].split("|")) for row in component_rows
    ) == 5


def test_contract_markdown_declares_version_and_decisions():
    contract = (ROOT / "artifacts" / "dryad_msg3d" / "DATA_CONTRACT_v1.md").read_text(
        encoding="utf-8"
    )
    assert CONTRACT_VERSION in contract
    for required in (
        "M = 2",
        "TARGET_T = 60",
        "PRESERVE_RAW_WITH_FLAG",
        "LEAKAGE_SAFE_COMPONENT_SPLIT",
        "SAFE_TO_BUILD_PREPROCESSING = YES",
    ):
        assert required in contract

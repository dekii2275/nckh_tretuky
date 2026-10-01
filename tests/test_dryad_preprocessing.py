"""Serialized tensor and coordinate-preservation checks for Round 3."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "dryad_msg3d"


def test_canonical_tensor_shape_dtype_person_and_sentinel():
    data = np.load(ARTIFACT_ROOT / "preprocessed" / "all_data.npy", mmap_mode="r")
    assert data.shape == (800, 3, 60, 25, 2)
    assert data.dtype == np.float32
    assert np.count_nonzero(data[:, :, :, :, 1]) == 0
    assert np.count_nonzero(data == 100.0) == 101
    assert np.isfinite(data).all()


def test_tensor_stats_record_contract_checks():
    stats = json.loads((ARTIFACT_ROOT / "tensor_stats.json").read_text(encoding="utf-8"))
    assert stats["serialization_exact_equality"] is True
    assert stats["nan_count"] == 0
    assert stats["inf_count"] == 0
    assert stats["sentinel_occurrences_valid_data"] == 101
    assert stats["raw_t_min"] == 21
    assert stats["raw_t_max"] == 60
    assert stats["roundtrip_max_abs_error_vs_float32_raw"] == 0.0
    assert stats["roundtrip_mean_abs_error_vs_float32_raw"] == 0.0
    assert stats["total_padded_frames"] == 19960

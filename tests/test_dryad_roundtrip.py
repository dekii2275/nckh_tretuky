"""Golden sample and inverse-layout checks recorded by the build."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "dryad_msg3d"


def test_golden_samples_cover_required_categories():
    rows = json.loads((ARTIFACT_ROOT / "golden_samples.json").read_text(encoding="utf-8"))
    categories = {row["golden_category"] for row in rows}
    assert categories == {
        "ASD original without sentinel",
        "ASD original with sentinel",
        "TD original",
        "ASD augmentation",
        "TD augmentation",
        "shortest-T sample",
        "longest-T sample",
    }
    for row in rows:
        assert row["raw_shape"][1:] == [25, 3]
        assert row["mapped_shape"] == row["raw_shape"]
        assert row["padded_shape"] == [60, 25, 3]
        assert row["final_shape"] == [3, 60, 25, 2]
        assert row["person1_nonzero_count"] == 0
        assert row["padded_region_nonzero_count"] == 0
        assert row["valid_region_max_abs_diff"] == 0.0
        assert row["max_abs_error_vs_float32_raw"] == 0.0


def test_sentinel_golden_sample_preserves_all_sentinels():
    rows = json.loads((ARTIFACT_ROOT / "golden_samples.json").read_text(encoding="utf-8"))
    sentinel_row = next(row for row in rows if row["sentinel_count"] > 0)
    assert sentinel_row["sentinel_count"] == 14
    assert sentinel_row["raw_sentinel_count"] == 14

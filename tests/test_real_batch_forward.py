"""Assertions for the forward-only real-batch smoke test."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "dryad_msg3d"


def test_real_batch_forward_artifact_is_finite_and_plain():
    result = json.loads(
        (ARTIFACT_ROOT / "real_batch_forward_smoke.json").read_text(encoding="utf-8")
    )
    assert result["protocol"] == "random_reproduction_like"
    assert result["batch_data_shape"] == [4, 3, 60, 25, 2]
    assert result["batch_label_shape"] == [4]
    assert result["person1_absmax"] == 0.0
    assert result["feature_shape"] == [4, 384]
    assert result["logits_shape"] == [4, 2]
    assert result["feature_finite"] is True
    assert result["logits_finite"] is True
    assert result["checkpoint_load"]["ignored_keys"] == ["fc.bias", "fc.weight"]
    assert result["checkpoint_load"]["unexpected_keys"] == []
    assert result["training_performed"] is False
    assert result["optimizer_step_performed"] is False

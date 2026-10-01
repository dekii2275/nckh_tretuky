"""Round 4 must be restricted to the audited single fold before training."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import run_round4_fold00 as runner  # noqa: E402


def test_round4_fold00_preflight_and_sentinel_exposure():
    rows = runner.read_manifest()
    fold = runner.read_fold()
    summary = runner.verify_fold(rows, fold)
    assert summary["train_sample_count"] == 720
    assert summary["test_sample_count"] == 80
    assert summary["duplicate_hash_overlap"] == 24
    exposure = runner.sentinel_exposure(rows, fold)
    assert exposure["train_subjects"] == ["ASD_004", "ASD_008", "ASD_022"]
    assert exposure["test_subjects"] == []
    assert exposure["train_sentinel_count"] == 101
    assert exposure["test_sentinel_count"] == 0

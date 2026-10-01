"""Static checks for the proposed Dryad -> model joint permutation."""

from pathlib import Path
import sys


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from dryad_joint_mapping import MODEL_JOINTS, MODEL_TO_RAW, RAW_JOINTS, RAW_TO_MODEL


def test_joint_mapping_is_a_bijection():
    assert len(RAW_JOINTS) == len(MODEL_JOINTS) == 25
    assert sorted(RAW_TO_MODEL) == list(range(25))
    assert sorted(MODEL_TO_RAW) == list(range(25))
    for raw_index, model_index in enumerate(RAW_TO_MODEL):
        assert MODEL_TO_RAW[model_index] == raw_index


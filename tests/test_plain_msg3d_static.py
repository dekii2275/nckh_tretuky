"""Static checks for the plain MSG3D path; no data loader or training."""

from pathlib import Path
import sys

import torch


AUTISM_ROOT = Path(__file__).resolve().parents[1] / "autism"
if str(AUTISM_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTISM_ROOT))

from plain_msg3d import build_plain_msg3d, load_ntu120_backbone


def test_plain_forward_has_two_class_logits_and_no_angle():
    model = build_plain_msg3d()
    assert model.dream is False
    assert model.use_angle is False
    assert model.data_bn.num_features == 2 * 3 * 25
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 21, 25, 2))
    assert tuple(output.shape) == (1, 2)


def test_plain_forward_accepts_another_temporal_length():
    model = build_plain_msg3d()
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 60, 25, 2))
    assert tuple(output.shape) == (1, 2)


def test_ntu120_backbone_load_excludes_classification_head():
    model = build_plain_msg3d()
    checkpoint = AUTISM_ROOT / "msg3d_pretrained_weights" / "ntu120-xsub-joint.pt"
    result = load_ntu120_backbone(model, checkpoint)
    assert result["ignored_keys"] == ["fc.bias", "fc.weight"]
    assert result["missing_keys"] == ["fc.weight", "fc.bias"]
    assert result["unexpected_keys"] == []

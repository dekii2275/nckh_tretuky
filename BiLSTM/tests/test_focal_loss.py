import torch
import torch.nn as nn
import pytest

from src.losses import FocalLoss


def test_focal_loss_gamma_zero_matches_cross_entropy():
    torch.manual_seed(42)
    inputs = torch.randn(16, 10)
    targets = torch.randint(0, 10, (16,))

    ce_loss = nn.CrossEntropyLoss(reduction="mean")(inputs, targets)
    focal_loss = FocalLoss(gamma=0.0, reduction="mean")(inputs, targets)

    assert torch.allclose(ce_loss, focal_loss, atol=1e-6)


def test_focal_loss_downweights_easy_examples():
    # Two samples: one easy (high confidence on correct class), one hard (low confidence)
    inputs = torch.tensor([
        [3.0, -3.0],  # easy for class 0 (prob ~0.9975)
        [0.1, -0.1],  # hard/ambiguous for class 0 (prob ~0.55)
    ])
    targets = torch.tensor([0, 0])

    ce = nn.CrossEntropyLoss(reduction="none")(inputs, targets)
    fl = FocalLoss(gamma=2.0, reduction="none")(inputs, targets)

    # Easy sample loss ratio should be drastically reduced compared to hard sample loss ratio
    ratio_easy = fl[0] / ce[0]
    ratio_hard = fl[1] / ce[1]

    assert ratio_easy < 0.01  # downweighted by > 100x
    assert ratio_hard > 0.20  # retains significant weight


def test_focal_loss_with_class_weights():
    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 4, (8,))
    weights = torch.tensor([1.0, 2.0, 0.5, 1.5])

    fl = FocalLoss(gamma=1.5, weight=weights, reduction="mean")(inputs, targets)
    assert fl > 0
    assert torch.isfinite(fl)

"""Small, dependency-light entry points for the plain MSG3D baseline.

The released training program also wires in Skepxel/ViT and an angle-aware
``dream`` branch.  This module exposes only the skeleton -> MSG3D -> FC path
needed for static baseline checks; it does not load or construct ViT.
"""

from collections import OrderedDict
from pathlib import Path

import torch

from model.msg3d import Model


def build_plain_msg3d(
    num_class=2,
    num_point=25,
    num_person=2,
    num_gcn_scales=13,
    num_g3d_scales=6,
    graph="graph.ntu_rgb_d.AdjMatrixGraph",
    in_channels=3,
):
    """Build plain MSG3D with no angle matrix and no auxiliary stream."""

    return Model(
        num_class=num_class,
        num_point=num_point,
        num_person=num_person,
        num_gcn_scales=num_gcn_scales,
        num_g3d_scales=num_g3d_scales,
        graph=graph,
        in_channels=in_channels,
        dream=False,
        use_angle=False,
    )


def load_ntu120_backbone(model, checkpoint_path, map_location="cpu"):
    """Load NTU120 weights while explicitly excluding the 120-class head."""

    checkpoint = torch.load(Path(checkpoint_path), map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a state-dict-like checkpoint")

    state = checkpoint.get("state_dict", checkpoint)
    state = OrderedDict((key.removeprefix("module."), value) for key, value in state.items())
    ignored_keys = {"fc.weight", "fc.bias"}
    filtered_state = OrderedDict(
        (key, value) for key, value in state.items() if key not in ignored_keys
    )
    incompatible = model.load_state_dict(filtered_state, strict=False)
    return {
        "ignored_keys": sorted(ignored_keys.intersection(state)),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "checkpoint_keys": len(state),
    }

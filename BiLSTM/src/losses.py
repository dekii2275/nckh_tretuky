"""Loss functions for MMASD action recognition."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class Focal Loss with optional class weighting.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t is the model's estimated probability for the ground-truth class.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        if weight is not None:
            self.register_buffer("weight", weight.clone().detach())
        else:
            self.weight = None
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: [B, C], targets: [B]
        log_prob = F.log_softmax(inputs, dim=-1)
        log_pt = log_prob.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        pt = torch.exp(log_pt)
        focal_weight = (1.0 - pt).clamp_min(0.0) ** self.gamma
        loss = -focal_weight * log_pt
        if self.weight is not None:
            class_w = self.weight[targets]
            loss = loss * class_w
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

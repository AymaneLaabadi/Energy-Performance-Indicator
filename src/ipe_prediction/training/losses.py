"""Training losses."""

from __future__ import annotations

import torch
from torch import nn

from ipe_prediction.training.preprocessing import TargetScaler


class StandardizedMSELoss(nn.Module):
    """Mean squared error in the standardised target space of :class:`TargetScaler`.

    Six terms, averaged with equal weights: ``HLC`` (computed as ``HLC_mit + HLC_ext``),
    ``HLC_mit``, ``HLC_ext`` (standardised in W/K) and the three time constants (standardised in
    log scale). Standardising puts W/K and hours on the same footing; the log on the time constants
    handles their right skew.
    """

    def __init__(self, target_scaler: TargetScaler) -> None:
        super().__init__()
        self.target_scaler = target_scaler

    def forward(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the scalar loss and the per-target MSE ``(6,)`` (detached, for logging)."""
        z_pred = self.target_scaler.outputs_to_standardized(outputs)
        z_true = self.target_scaler.encode(targets)
        per_target = (z_pred - z_true).pow(2).mean(0)
        return per_target.mean(), per_target.detach()

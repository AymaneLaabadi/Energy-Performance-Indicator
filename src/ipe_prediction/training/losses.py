"""Training loss."""

from __future__ import annotations

import torch

from ipe_prediction.training.preprocessing import Preprocessor


def standardized_mse(
    out: torch.Tensor, targets: torch.Tensor, prep: Preprocessor
) -> tuple[torch.Tensor, torch.Tensor]:
    """MSE in the standardised target space, averaged over the 6 targets with equal weights.

    Returns the scalar loss and the per-target MSE ``(6,)`` (detached, for logging).
    """
    per_target = (prep.standardized(out.float()) - prep.encode(targets)).pow(2).mean(0)
    return per_target.mean(), per_target.detach()

"""Training losses, in the standardised target space of the preprocessor."""

from __future__ import annotations

import torch

from ipe_prediction.training.preprocessing import Preprocessor

LOSSES = ("mse", "gaussian_nll")
#: beta of the beta-NLL (Seitzer et al., 2022): each sample's NLL is weighted by var**beta
#: (no gradient), which keeps the mean as well fitted as with an MSE.
NLL_BETA: float = 0.5


def n_network_outputs(prep: Preprocessor, loss: str) -> int:
    """Outputs of the network: the means, plus one log-variance each with the Gaussian NLL."""
    return prep.n_outputs * (2 if loss == "gaussian_nll" else 1)


def target_loss(
    out: torch.Tensor, targets: torch.Tensor, prep: Preprocessor, loss: str = "mse"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Training objective and per-target MSE of the means ``(T,)`` (detached, for logging).

    * ``"mse"``: mean squared error, averaged over the targets with equal weights;
    * ``"gaussian_nll"``: beta-NLL of a Gaussian per target (``out`` = means then log-variances).
    """
    out = out.float()
    mean = out[:, : prep.n_outputs]
    sq = (prep.standardized(mean) - prep.encode(targets)).pow(2)
    per_target = sq.mean(0).detach()
    if loss == "mse":
        return sq.mean(0).mean(), per_target
    var = prep.standardized_variance(out[:, prep.n_outputs :].clamp(-10, 10).exp())
    nll = 0.5 * (var.log() + sq / var)
    return (var.detach() ** NLL_BETA * nll).mean(), per_target

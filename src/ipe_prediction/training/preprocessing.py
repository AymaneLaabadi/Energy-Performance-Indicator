"""Input and target scaling, fitted on the training split only.

* Input: ``delta_t / std(delta_t)``; the heater series stays binary (0/1).
* Targets: the heat-loss coefficients (``HLC``, ``HLC_mit``, ``HLC_ext``) are standardised in
  their natural scale (W/K); the time constants (``tau_b``, ``tau_n``, ``tau_inf``) are
  log-transformed then standardised (they are strongly right-skewed, see the EDA).

The scaler is an ``nn.Module`` holding its statistics as buffers, so it is saved with the model.
"""

from __future__ import annotations

import torch
from torch import nn


def fit_delta_t_std(delta_t: torch.Tensor, indices: torch.Tensor, chunk: int = 50_000) -> float:
    """Std of all ``delta_t`` values over the given rows, computed in chunks (float64)."""
    s = s2 = 0.0
    n = 0
    for i in range(0, len(indices), chunk):
        x = delta_t[indices[i : i + chunk]].double()
        s += float(x.sum())
        s2 += float(x.pow(2).sum())
        n += x.numel()
    mean = s / n
    return (s2 / n - mean**2) ** 0.5


class TargetScaler(nn.Module):
    """Map between physical targets and the standardised space used by the network and loss.

    Physical order (6): ``[HLC, HLC_mit, HLC_ext, tau_b, tau_n, tau_inf]``.
    Network output order (5): ``[HLC_mit, HLC_ext, tau_b, tau_n, tau_inf]`` in standardised
    space; ``HLC`` is not predicted but computed as ``HLC_mit + HLC_ext``.
    """

    def __init__(
        self,
        lin_mean: torch.Tensor,
        lin_std: torch.Tensor,
        log_mean: torch.Tensor,
        log_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("lin_mean", lin_mean.float())  # HLC, HLC_mit, HLC_ext
        self.register_buffer("lin_std", lin_std.float())
        self.register_buffer("log_mean", log_mean.float())  # log tau_b, log tau_n, log tau_inf
        self.register_buffer("log_std", log_std.float())

    @classmethod
    def fit(cls, targets: torch.Tensor) -> TargetScaler:
        """Fit the statistics on ``(N, 6)`` training targets."""
        lin = targets[:, :3].double()
        log = targets[:, 3:].double().log()
        return cls(lin.mean(0), lin.std(0), log.mean(0), log.std(0))

    @classmethod
    def empty(cls) -> TargetScaler:
        """Placeholder to be filled by ``load_state_dict``."""
        return cls(torch.zeros(3), torch.ones(3), torch.zeros(3), torch.ones(3))

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        """``(B, 6)`` physical targets -> ``(B, 6)`` standardised targets."""
        z_lin = (y[:, :3] - self.lin_mean) / self.lin_std
        z_log = (y[:, 3:].log() - self.log_mean) / self.log_std
        return torch.cat([z_lin, z_log], dim=1)

    def outputs_to_standardized(self, out: torch.Tensor) -> torch.Tensor:
        """``(B, 5)`` network outputs -> ``(B, 6)`` standardised predictions (adds HLC)."""
        mit = self.lin_mean[1] + self.lin_std[1] * out[:, 0]
        ext = self.lin_mean[2] + self.lin_std[2] * out[:, 1]
        z_hlc = (mit + ext - self.lin_mean[0]) / self.lin_std[0]
        return torch.cat([z_hlc[:, None], out], dim=1)

    def decode(self, out: torch.Tensor) -> torch.Tensor:
        """``(B, 5)`` network outputs -> ``(B, 6)`` physical predictions."""
        mit = self.lin_mean[1] + self.lin_std[1] * out[:, 0]
        ext = self.lin_mean[2] + self.lin_std[2] * out[:, 1]
        taus = (self.log_mean + self.log_std * out[:, 2:]).exp()
        return torch.cat([(mit + ext)[:, None], mit[:, None], ext[:, None], taus], dim=1)

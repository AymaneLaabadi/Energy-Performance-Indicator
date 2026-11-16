"""Model inputs and target scaling, fitted on the training split only.

Inputs ``(B, C, L)``: ``delta_t / std(delta_t)`` and ``heater_on``. With ``extra_features``,
three more channels: elapsed time ``t / L``, cumulative heating ``cumsum(heater_on) / L`` and the
integral of the rise ``cumsum(delta_t / std) / L``. Integrated over ``[0, t]``, the lumped
energy balance ``C dT = U dt - H T dt`` is linear in those channels.

Targets: ``HLC``, ``HLC_mit``, ``HLC_ext`` are standardised in W/K; the time constants are
log-transformed (right-skewed) then standardised. Networks output 5 standardised values
``[HLC_mit, HLC_ext, log tau_b, log tau_n, log tau_inf]``; ``HLC`` is always ``HLC_mit +
HLC_ext``, so the additivity constraint holds by construction.
"""

from __future__ import annotations

import torch
from torch import nn

#: Number of network outputs (all targets but HLC, which is their mit + ext sum).
N_OUTPUTS: int = 5
_LOG = torch.tensor([False, False, False, True, True, True])  # targets in log space


def n_inputs(extra_features: bool) -> int:
    """Number of input channels built by :meth:`Preprocessor.inputs`."""
    return 5 if extra_features else 2


class Preprocessor(nn.Module):
    """Scaling statistics (buffers, saved in the checkpoint) and the input / target transforms."""

    def __init__(self, extra_features: bool = False) -> None:
        super().__init__()
        self.extra_features = extra_features
        self.register_buffer("delta_t_std", torch.ones(()))
        self.register_buffer("mean", torch.zeros(6))  # of the (log-)targets, canonical order
        self.register_buffer("std", torch.ones(6))
        self.register_buffer("log", _LOG.clone(), persistent=False)  # follows .to(device)

    @classmethod
    def fit(
        cls,
        delta_t: torch.Tensor,
        targets: torch.Tensor,
        indices: torch.Tensor,
        extra_features: bool = False,
        chunk: int = 50_000,
    ) -> Preprocessor:
        """Fit on the rows ``indices`` (the training split); ``delta_t`` is read in chunks."""
        prep = cls(extra_features)
        s = s2 = n = 0.0
        for i in range(0, len(indices), chunk):
            x = delta_t[indices[i : i + chunk]].double()
            s, s2, n = s + float(x.sum()), s2 + float(x.pow(2).sum()), n + x.numel()
        prep.delta_t_std.fill_((s2 / n - (s / n) ** 2) ** 0.5)
        y = prep._forward_transform(targets[indices].double())
        prep.mean.copy_(y.mean(0))
        prep.std.copy_(y.std(0))
        return prep

    def inputs(self, delta_t: torch.Tensor, heater_on: torch.Tensor) -> torch.Tensor:
        """``(B, L)`` rise in K and ``(B, L)`` heater state -> ``(B, C, L)`` network input."""
        x = delta_t / self.delta_t_std
        channels = [x, heater_on]
        if self.extra_features:
            length = x.shape[1]
            t = torch.linspace(0.0, 1.0, length, device=x.device).expand_as(x)
            channels += [t, heater_on.cumsum(1) / length, x.cumsum(1) / length]
        return torch.stack(channels, dim=1)

    def _forward_transform(self, y: torch.Tensor) -> torch.Tensor:
        return torch.where(self.log, y.log(), y)

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        """``(B, 6)`` physical targets -> ``(B, 6)`` standardised targets."""
        return (self._forward_transform(y) - self.mean) / self.std

    def standardized(self, out: torch.Tensor) -> torch.Tensor:
        """``(B, 5)`` network outputs -> ``(B, 6)`` standardised predictions (adds HLC)."""
        mit, ext = (self.mean[1:3] + self.std[1:3] * out[:, :2]).unbind(1)
        z_hlc = (mit + ext - self.mean[0]) / self.std[0]
        return torch.cat([z_hlc[:, None], out], dim=1)

    def decode(self, out: torch.Tensor) -> torch.Tensor:
        """``(B, 5)`` network outputs -> ``(B, 6)`` physical predictions."""
        y = self.mean + self.std * self.standardized(out)
        return torch.where(self.log, y.exp(), y)

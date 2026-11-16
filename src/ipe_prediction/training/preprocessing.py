"""Model inputs and target scaling, fitted on the training split only.

Inputs ``(B, C, L)``: ``delta_t / std(delta_t)`` and ``heater_on``. With ``extra_features``,
three more channels: elapsed time ``t / L``, cumulative heating ``cumsum(heater_on) / L`` and the
integral of the rise ``cumsum(delta_t / std) / L``. Integrated over ``[0, t]``, the lumped
energy balance ``C dT = U dt - H T dt`` is linear in those channels.

Targets: a selection of ``data.TARGET_NAMES`` (all 6 by default), kept in canonical order. The
heat-loss coefficients are standardised in W/K; the time constants are log-transformed
(right-skewed) then standardised. The network predicts every selected target directly, except
``HLC`` when ``HLC``, ``HLC_mit`` and ``HLC_ext`` are all selected: it is then computed as
``HLC_mit + HLC_ext``, so the additivity constraint holds by construction. With all 6 targets the
network has 5 outputs ``[HLC_mit, HLC_ext, log tau_b, log tau_n, log tau_inf]``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from ipe_prediction.data import TARGET_NAMES

#: Network outputs with all targets (HLC is their mit + ext sum).
N_OUTPUTS: int = 5
_LOG_TARGETS = ("tau_b", "tau_n", "tau_inf")  # standardised in log space
_HB = ("HLC", "HLC_mit", "HLC_ext")


def n_inputs(extra_features: bool) -> int:
    """Number of input channels built by :meth:`Preprocessor.inputs`."""
    return 5 if extra_features else 2


class Preprocessor(nn.Module):
    """Scaling statistics (buffers, saved in the checkpoint) and the input / target transforms.

    Attributes:
        target_names: selected targets, canonical order.
        derive_hlc: True when HLC is computed as ``HLC_mit + HLC_ext``.
        n_outputs: number of network outputs.
    """

    def __init__(self, extra_features: bool = False, targets: Sequence[str] = TARGET_NAMES) -> None:
        super().__init__()
        unknown = set(targets) - set(TARGET_NAMES)
        if unknown or not targets:
            raise ValueError(f"targets must be a non-empty subset of {TARGET_NAMES}: {targets}")
        self.extra_features = extra_features
        self.target_names = tuple(n for n in TARGET_NAMES if n in targets)
        self.derive_hlc = set(_HB) <= set(self.target_names)
        self.n_outputs = len(self.target_names) - self.derive_hlc
        columns = [TARGET_NAMES.index(n) for n in self.target_names]
        n = len(columns)
        self.register_buffer("delta_t_std", torch.ones(()))
        self.register_buffer("mean", torch.zeros(n))  # of the selected (log-)targets
        self.register_buffer("std", torch.ones(n))
        # constants rebuilt from the config, kept out of the checkpoint; follow .to(device)
        self.register_buffer("columns", torch.tensor(columns), persistent=False)
        is_log = [name in _LOG_TARGETS for name in self.target_names]
        self.register_buffer("log", torch.tensor(is_log), persistent=False)

    @classmethod
    def fit(
        cls,
        delta_t: torch.Tensor,
        targets: torch.Tensor,
        indices: torch.Tensor,
        extra_features: bool = False,
        target_names: Sequence[str] = TARGET_NAMES,
        chunk: int = 50_000,
    ) -> Preprocessor:
        """Fit on the rows ``indices`` (the training split); ``delta_t`` is read in chunks."""
        prep = cls(extra_features, target_names)
        s = s2 = n = 0.0
        for i in range(0, len(indices), chunk):
            x = delta_t[indices[i : i + chunk]].double()
            s, s2, n = s + float(x.sum()), s2 + float(x.pow(2).sum()), n + x.numel()
        prep.delta_t_std.fill_((s2 / n - (s / n) ** 2) ** 0.5)
        y = prep._forward_transform(prep.select(targets[indices]).double())
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

    def select(self, y: torch.Tensor) -> torch.Tensor:
        """``(B, 6)`` all targets -> ``(B, T)`` the selected ones."""
        return y[:, self.columns.to(y.device)]

    def _forward_transform(self, y: torch.Tensor) -> torch.Tensor:
        return torch.where(self.log.to(y.device), y.log(), y)

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        """``(B, 6)`` physical targets -> ``(B, T)`` standardised selected targets."""
        return (self._forward_transform(self.select(y)) - self.mean) / self.std

    def standardized(self, out: torch.Tensor) -> torch.Tensor:
        """``(B, n_outputs)`` network outputs -> ``(B, T)`` standardised predictions."""
        if not self.derive_hlc:
            return out
        mit, ext = (self.mean[1:3] + self.std[1:3] * out[:, :2]).unbind(1)  # HLC is column 0
        z_hlc = (mit + ext - self.mean[0]) / self.std[0]
        return torch.cat([z_hlc[:, None], out], dim=1)

    def standardized_variance(self, var: torch.Tensor) -> torch.Tensor:
        """``(B, n_outputs)`` predicted variances -> ``(B, T)``, adding that of the derived HLC.

        HLC = HLC_mit + HLC_ext, with the two errors assumed independent.
        """
        if not self.derive_hlc:
            return var
        var_hlc = (self.std[1] ** 2 * var[:, 0] + self.std[2] ** 2 * var[:, 1]) / self.std[0] ** 2
        return torch.cat([var_hlc[:, None], var], dim=1)

    def from_standardized(self, z: torch.Tensor) -> torch.Tensor:
        """``(B, T)`` standardised values -> physical units (exp for the time constants)."""
        y = self.mean + self.std * z
        return torch.where(self.log, y.exp(), y)

    def decode(self, out: torch.Tensor) -> torch.Tensor:
        """``(B, n_outputs)`` network outputs (means) -> ``(B, T)`` physical predictions."""
        return self.from_standardized(self.standardized(out))

"""Baseline: convolutional front end + bidirectional GRU."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ipe_prediction.training.preprocessing import N_OUTPUTS


@dataclass
class CNNGRUConfig:
    """Hyper-parameters of :class:`CNNGRU` (one entry per conv layer in the ``conv_*`` tuples)."""

    conv_channels: tuple[int, ...] = (32, 64)
    conv_kernels: tuple[int, ...] = (5, 5)
    conv_strides: tuple[int, ...] = (2, 2)
    gru_hidden: int = 32
    gru_layers: int = 1
    mlp_hidden: int = 32


class CNNGRU(nn.Module):
    """``(B, C_in, L) -> (B, 5)``.

    Conv1d + GELU layers (strided: 480 -> 60 steps by default), a bidirectional GRU, and an MLP
    on the final forward and backward states of the top layer.
    """

    def __init__(self, cfg: CNNGRUConfig, in_channels: int, n_outputs: int = N_OUTPUTS) -> None:
        super().__init__()
        layers, c_in = [], in_channels
        for c_out, k, s in zip(cfg.conv_channels, cfg.conv_kernels, cfg.conv_strides, strict=True):
            layers += [nn.Conv1d(c_in, c_out, k, stride=s, padding=k // 2), nn.GELU()]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        self.gru = nn.GRU(
            c_in, cfg.gru_hidden, cfg.gru_layers, batch_first=True, bidirectional=True
        )
        self.head = nn.Sequential(
            nn.Linear(2 * cfg.gru_hidden, cfg.mlp_hidden),
            nn.GELU(),
            nn.Linear(cfg.mlp_hidden, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).transpose(1, 2)  # (B, L', C)
        _, h_n = self.gru(h)  # (layers * 2, B, H)
        return self.head(torch.cat([h_n[-2], h_n[-1]], dim=1))

"""Model A: residual convolutional encoder with average pooling, then a bidirectional GRU."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ipe_prediction.training.preprocessing import N_OUTPUTS


@dataclass
class ModelAConfig:
    """Hyper-parameters of :class:`ModelA`.

    The encoder has one stage per ``stage_channels`` entry: ``blocks_per_stage`` residual blocks
    then an average pooling by ``pool``. With 3 stages and ``pool=2`` the GRU sees 480 / 8 = 60
    steps of 8 minutes.
    """

    stem_channels: int = 32
    stage_channels: tuple[int, ...] = (32, 64, 128)
    blocks_per_stage: int = 2
    kernel: int = 5
    pool: int = 2
    gru_hidden: int = 64
    gru_layers: int = 2
    dropout: float = 0.1
    mlp_hidden: int = 64


class ResidualBlock1d(nn.Module):
    """ResNet basic block in 1D: ``ReLU(BN(Conv(ReLU(BN(Conv(x))))) + shortcut(x))``.

    The shortcut is the identity, or a 1x1 convolution + BN when the number of channels changes.
    """

    def __init__(self, c_in: int, c_out: int, kernel: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(c_out),
            nn.ReLU(),
            nn.Conv1d(c_out, c_out, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(c_out),
        )
        self.shortcut = (
            nn.Identity()
            if c_in == c_out
            else nn.Sequential(nn.Conv1d(c_in, c_out, 1, bias=False), nn.BatchNorm1d(c_out))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.body(x) + self.shortcut(x))


class ModelA(nn.Module):
    """``(B, C_in, L) -> (B, 5)``.

    Stem conv, residual stages each followed by average pooling (480 -> 60 steps by default), a
    bidirectional GRU over the pooled sequence, and an MLP on the final forward and backward
    states of the top GRU layer.
    """

    def __init__(self, cfg: ModelAConfig, in_channels: int, n_outputs: int = N_OUTPUTS) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, cfg.stem_channels, 7, padding=3, bias=False),
            nn.BatchNorm1d(cfg.stem_channels),
            nn.ReLU(),
        ]
        c = cfg.stem_channels
        for c_out in cfg.stage_channels:
            for i in range(cfg.blocks_per_stage):
                layers.append(ResidualBlock1d(c if i == 0 else c_out, c_out, cfg.kernel))
            layers.append(nn.AvgPool1d(cfg.pool))
            c = c_out
        self.encoder = nn.Sequential(*layers)
        self.gru = nn.GRU(
            c,
            cfg.gru_hidden,
            cfg.gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.gru_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(2 * cfg.gru_hidden, cfg.mlp_hidden),
            nn.GELU(),
            nn.Linear(cfg.mlp_hidden, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x).transpose(1, 2)  # (B, L', C)
        _, h_n = self.gru(h)  # (layers * 2, B, H)
        return self.head(torch.cat([h_n[-2], h_n[-1]], dim=1))

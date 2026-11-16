"""Dilated residual TCN + multi-head attention pooling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ipe_prediction.training.preprocessing import N_OUTPUTS


@dataclass
class TCNAttentionConfig:
    """Hyper-parameters of :class:`TCNAttention`.

    With kernel 3 and stem stride 2, the receptive field is ``1 + 4 * sum(dilations)`` = 509
    steps of 2 min: the whole 8 h window.
    """

    channels: int = 64
    kernel: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    stem_stride: int = 2
    dropout: float = 0.1
    heads: int = 4
    mlp_hidden: int = 64


class ResidualBlock(nn.Module):
    """``x + Conv(Drop(GELU(BN(Conv(GELU(BN(x)))))))`` with 'same' dilated convolutions.

    BatchNorm rather than a per-sample norm, which would erase the amplitude of the rise.
    """

    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.body = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class AttentionPooling(nn.Module):
    """Multi-head additive attention over time: ``(B, C, T) -> (B, C)``.

    Each head scores every step, takes a softmax over time and averages its slice of a value
    projection; the heads are concatenated. The softmax runs in float32 (safe under autocast).
    """

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError(f"heads={heads} must divide channels={channels}")
        self.heads = heads
        self.score = nn.Sequential(
            nn.Linear(channels, channels // 2), nn.Tanh(), nn.Linear(channels // 2, heads)
        )
        self.value = nn.Linear(channels, channels)

    def weights(self, x: torch.Tensor) -> torch.Tensor:
        """Attention weights ``(B, T, heads)``, summing to 1 over time."""
        return self.score(x.transpose(1, 2)).float().softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.weights(x)  # (B, T, heads)
        v = self.value(x.transpose(1, 2)).float().unflatten(-1, (self.heads, -1))  # (B, T, h, d)
        return torch.einsum("bth,bthd->bhd", a, v).flatten(1).to(x.dtype)


class TCNAttention(nn.Module):
    """``(B, C_in, L) -> (B, 5)``.

    Strided stem conv (480 -> 240 steps), residual dilated blocks, attention pooling, MLP head.
    """

    def __init__(self, cfg: TCNAttentionConfig, in_channels: int) -> None:
        super().__init__()
        c, s = cfg.channels, cfg.stem_stride
        self.stem = nn.Conv1d(in_channels, c, kernel_size=2 * s + 1, stride=s, padding=s)
        self.blocks = nn.Sequential(
            *(ResidualBlock(c, cfg.kernel, d, cfg.dropout) for d in cfg.dilations),
            nn.BatchNorm1d(c),
            nn.GELU(),
        )
        self.pool = AttentionPooling(c, cfg.heads)
        self.head = nn.Sequential(
            nn.Linear(c, cfg.mlp_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden, N_OUTPUTS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.blocks(self.stem(x))))

    @torch.no_grad()
    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Attention weights ``(B, T, heads)`` on the stem's time grid (for interpretation)."""
        return self.pool.weights(self.blocks(self.stem(x)))

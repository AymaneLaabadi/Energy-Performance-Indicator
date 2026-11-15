"""Descriptive helpers for the exploratory analysis (no fitting of any predictor).

All functions are vectorised over simulations: series have shape ``(N, L)``.
"""

from __future__ import annotations

import torch


def excitation_descriptors(power: torch.Tensor) -> dict[str, torch.Tensor]:
    """Describe the heating signal ``U(t)`` of every simulation.

    Args:
        power: ``(N, L)`` heating power in W.

    Returns:
        Mapping with ``duty_cycle`` (share of time at full power, in [0, 1]) and
        ``n_switches`` (number of on/off transitions), each of shape ``(N,)``.
    """
    p_max = power.amax(1).clamp(min=1.0)
    return {
        "duty_cycle": power.mean(1) / p_max,
        "n_switches": (power[:, 1:] != power[:, :-1]).sum(1).float(),
    }


def downsample(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Average-pool ``(N, L)`` series by ``factor`` (``L`` must be divisible by ``factor``)."""
    if x.shape[1] % factor:
        raise ValueError(f"length {x.shape[1]} is not divisible by {factor}")
    return x.reshape(x.shape[0], -1, factor).mean(2)


def pca(x: torch.Tensor, n_components: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Principal component analysis via SVD.

    Args:
        x: ``(N, F)`` data matrix.
        n_components: Number of components to return.

    Returns:
        ``(mean (F,), components (n_components, F), explained_variance_ratio (min(N, F),))``.
    """
    mean = x.mean(0)
    _, s, vh = torch.linalg.svd(x - mean, full_matrices=False)
    var = s.pow(2)
    return mean, vh[:n_components], var / var.sum()

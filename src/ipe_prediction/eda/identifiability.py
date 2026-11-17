"""Model-free identifiability analysis on groups of simulations that received the same input.

The 1M simulations use few distinct heating sequences (the PRBS generator always used the same
seed), so many simulations share exactly the same input ``U(t)``. Within such a group, the
temperature curves differ only because of the building (the targets) and of the hidden initial
state: a controlled experiment that needs no trained model.

All functions work on torch tensors and run on the GPU when one is available.
"""

from __future__ import annotations

import numpy as np
import torch

#: Columns of :func:`target_view`: heat-loss coefficients in W/K, time constants and the
#: effective capacity ``C_n = HLC x tau_n`` in log scale.
VIEW_NAMES: tuple[str, ...] = (
    "HLC",
    "HLC_mit",
    "HLC_ext",
    "log tau_b",
    "log tau_n",
    "log tau_inf",
    "log C_n",
)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def input_groups(heater_on: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Group simulations by identical heater sequence.

    Args:
        heater_on: ``(N, L)`` 0/1 heater state.

    Returns:
        ``group`` ``(N,)`` group index of each simulation (groups sorted by decreasing size) and
        ``size`` ``(G,)`` number of simulations per group.
    """
    packed = torch.cat(
        [
            torch.from_numpy(np.packbits(c.numpy().astype(np.uint8), axis=1))
            for c in heater_on.split(200_000)
        ]
    )
    _, group, size = torch.unique(packed, dim=0, return_inverse=True, return_counts=True)
    order = size.argsort(descending=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(len(order))
    return rank[group], size[order]


def target_view(targets: torch.Tensor) -> torch.Tensor:
    """``(N, 6)`` targets (canonical order) -> ``(N, 7)`` analysis view (see :data:`VIEW_NAMES`)."""
    y = targets.double()
    return torch.cat([y[:, :3], y[:, 3:].log(), (y[:, 0] * y[:, 4]).log()[:, None]], dim=1)


def ridge_r2(
    x: torch.Tensor, y: torch.Tensor, train_frac: float = 0.8, ridge: float = 1e-2, seed: int = 0
) -> torch.Tensor:
    """Held-out R² of a ridge regression of each column of ``y`` on ``x``.

    A linear lower bound on how much of each target the input determines. Features are
    standardised on the training part; ``ridge`` is relative to the number of samples.

    Args:
        x: ``(n, p)`` features (e.g. a downsampled temperature curve).
        y: ``(n, k)`` targets.

    Returns:
        ``(k,)`` R² on the held-out ``1 - train_frac`` share.
    """
    x, y = x.double().to(_DEVICE), y.double().to(_DEVICE)
    n = len(x)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).to(_DEVICE)
    tr, te = perm[: int(train_frac * n)], perm[int(train_frac * n) :]
    mx, sx = x[tr].mean(0), x[tr].std(0).clamp_min(1e-12)
    my = y[tr].mean(0)
    xt, xe = (x[tr] - mx) / sx, (x[te] - mx) / sx
    gram = xt.T @ xt + ridge * len(tr) * torch.eye(x.shape[1], dtype=x.dtype, device=_DEVICE)
    w = torch.linalg.solve(gram, xt.T @ (y[tr] - my))
    resid = y[te] - (xe @ w + my)
    return (1 - resid.pow(2).mean(0) / y[te].var(0)).cpu()


def knn_forward_r2(y: torch.Tensor, curves: torch.Tensor, k: int = 20) -> torch.Tensor:
    """How much of the curve the building explains, at each time step.

    Leave-one-out k-nearest-neighbour regression of the curve on the standardised targets: the
    curve of a simulation is predicted by the mean curve of the ``k`` simulations with the most
    similar targets (same input group). What remains is due to the hidden initial state.

    Args:
        y: ``(n, k_y)`` targets of the group.
        curves: ``(n, T)`` temperature rise of the group.

    Returns:
        ``(T,)`` R² of the curve value at each time step.
    """
    y, curves = y.double().to(_DEVICE), curves.double().to(_DEVICE)
    z = (y - y.mean(0)) / y.std(0).clamp_min(1e-12)
    pred = []
    for chunk in torch.arange(len(z), device=_DEVICE).split(1000):
        d = torch.cdist(z[chunk], z)
        d[torch.arange(len(chunk)), chunk] = float("inf")
        pred.append(curves[d.topk(k, largest=False).indices].mean(1))
    resid = curves - torch.cat(pred)
    return (1 - resid.pow(2).mean(0) / curves.var(0).clamp_min(1e-12)).cpu()


def curve_twins(
    curves: torch.Tensor, n_query: int = 2000, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest other curve (Euclidean, same group) for ``n_query`` random simulations.

    Returns:
        ``query`` ``(m,)`` and ``twin`` ``(m,)`` positions within ``curves``.
    """
    curves = curves.double().to(_DEVICE)
    g = torch.Generator().manual_seed(seed)
    query = torch.randperm(len(curves), generator=g)[: min(n_query, len(curves))].to(_DEVICE)
    twins = []
    for chunk in query.split(500):
        d = torch.cdist(curves[chunk], curves)
        d[torch.arange(len(chunk)), chunk] = float("inf")
        twins.append(d.argmin(1))
    return query.cpu(), torch.cat(twins).cpu()


def correlation_over_time(curves: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pearson correlation between the curve value at each time and each target ``(T, k)``."""
    c = curves.double() - curves.double().mean(0)
    t = y.double() - y.double().mean(0)
    cov = c.T @ t / len(c)
    return cov / (c.std(0, unbiased=False)[:, None] * t.std(0, unbiased=False)[None, :]).clamp_min(
        1e-12
    )

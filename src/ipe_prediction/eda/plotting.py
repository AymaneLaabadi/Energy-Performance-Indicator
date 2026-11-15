"""Matplotlib helpers for the exploratory analysis.

A single, consistent visual system is used: a fixed categorical order, one-hue sequential
ramps for magnitudes, a blue/orange diverging map with a neutral midpoint for correlations,
thin marks and recessive axes.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap

from ipe_prediction.data import TARGET_NAMES, TARGET_UNITS

#: Categorical hues, used in this fixed order (never cycled).
SERIES: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
)
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

#: Sequential (single hue, light -> dark) and diverging (blue - neutral - orange) maps.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_orange", ["#104281", "#3987e5", "#f0efec", "#ec835a", "#b8431a"]
)

#: Fixed colour per target family, so a target keeps its colour in every figure.
TARGET_COLORS: dict[str, str] = {
    "HLC": SERIES[0],
    "HLC_mit": SERIES[1],
    "HLC_ext": SERIES[2],
    "tau_b": SERIES[6],
    "tau_n": SERIES[3],
    "tau_inf": SERIES[4],
}


def set_style() -> None:
    """Apply the project-wide Matplotlib style."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.5,
            "figure.dpi": 110,
            "axes.prop_cycle": mpl.cycler(color=list(SERIES)),
        }
    )


def _np(x: torch.Tensor | np.ndarray) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def label(name: str) -> str:
    """Axis label ``name [unit]`` for a target."""
    return f"{name} [{TARGET_UNITS.get(name, '-')}]"


def target_histograms(targets: torch.Tensor, bins: int = 120) -> plt.Figure:
    """One histogram per target, with median and 1/99 % quantiles marked."""
    y = _np(targets)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2))
    for j, (ax, name) in enumerate(zip(axes.flat, TARGET_NAMES, strict=True)):
        col = y[:, j]
        ax.hist(col, bins=bins, color=TARGET_COLORS[name], alpha=0.85, edgecolor=SURFACE, lw=0.3)
        for q, ls in ((0.01, ":"), (0.5, "-"), (0.99, ":")):
            ax.axvline(np.quantile(col, q), color=INK_2, lw=0.9, ls=ls)
        ax.set_title(name)
        ax.set_xlabel(label(name))
        ax.set_ylabel("count")
    fig.suptitle("Target distributions (solid: median, dotted: 1 % / 99 %)", fontsize=12)
    fig.tight_layout()
    return fig


def correlation_heatmap(
    frame: pd.DataFrame, title: str, figsize: tuple[float, float] = (6.5, 5.5)
) -> plt.Figure:
    """Annotated correlation heatmap on a symmetric diverging scale."""
    corr = frame.to_numpy()
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(frame.shape[1]), frame.columns, rotation=40, ha="right")
    ax.set_yticks(range(frame.shape[0]), frame.index)
    ax.grid(False)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            v = corr[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="#ffffff" if abs(v) > 0.6 else INK,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def hexbin_pairs(
    targets: torch.Tensor,
    pairs: Sequence[tuple[str, str]],
    gridsize: int = 60,
    ncols: int = 3,
) -> plt.Figure:
    """Density (hexbin) scatter plots for selected target pairs."""
    y = _np(targets)
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.6 * nrows), squeeze=False)
    for ax, (a, b) in zip(axes.flat, pairs, strict=False):
        xa, xb = y[:, TARGET_NAMES.index(a)], y[:, TARGET_NAMES.index(b)]
        ax.hexbin(xa, xb, gridsize=gridsize, cmap=SEQUENTIAL, mincnt=1, bins="log", lw=0)
        r = np.corrcoef(xa, xb)[0, 1]
        ax.set_title(f"{b} vs {a}  (r = {r:.2f})")
        ax.set_xlabel(label(a))
        ax.set_ylabel(label(b))
    for ax in list(axes.flat)[len(pairs) :]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def sample_series(
    time_h: torch.Tensor,
    temperature: torch.Tensor,
    power: torch.Tensor,
    targets: torch.Tensor,
    indices: Sequence[int],
) -> plt.Figure:
    """Plot ``Tint`` (top) and ``U`` (bottom) for a few simulations, with their labels."""
    t, temp, pw, y = _np(time_h), _np(temperature), _np(power), _np(targets)
    n = len(indices)
    fig, axes = plt.subplots(
        2, n, figsize=(3.4 * n, 4.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )
    for k, i in enumerate(indices):
        ax_t, ax_u = axes[0, k], axes[1, k]
        ax_t.plot(t, temp[i], color=SERIES[0], lw=1.6)
        ax_u.fill_between(t, pw[i] / 1000, step="post", color=SERIES[1], alpha=0.35, lw=0)
        ax_u.step(t, pw[i] / 1000, where="post", color=SERIES[1], lw=1.0)
        ax_t.set_title(
            f"sim {i}\nHLC={y[i, 0]:.0f}  mit/ext={y[i, 1]:.0f}/{y[i, 2]:.0f} W/K\n"
            f"tau b/n/inf={y[i, 3]:.0f}/{y[i, 4]:.1f}/{y[i, 5]:.0f} h",
            fontsize=8,
            fontweight="normal",
        )
        ax_u.set_xlabel("time [h]")
        if k == 0:
            ax_t.set_ylabel("Tint [degC]")
            ax_u.set_ylabel("U [kW]")
        ax_u.set_ylim(-0.3, 7.6)
    fig.tight_layout()
    return fig


def curves_colored_by(
    time_h: torch.Tensor,
    curves: torch.Tensor,
    values: torch.Tensor,
    name: str,
    title: str,
    max_curves: int = 400,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Overlay curves coloured by a continuous value (sequential ramp, dark = high)."""
    t, c, v = _np(time_h), _np(curves), _np(values)
    if len(c) > max_curves:
        sel = np.random.default_rng(0).choice(len(c), max_curves, replace=False)
        c, v = c[sel], v[sel]
    order = np.argsort(v)
    norm = mpl.colors.Normalize(vmin=np.quantile(v, 0.02), vmax=np.quantile(v, 0.98))
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4))
    for i in order:
        ax.plot(t, c[i], color=SEQUENTIAL(norm(v[i])), lw=0.8, alpha=0.8)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=SEQUENTIAL)
    ax.figure.colorbar(sm, ax=ax, label=label(name), fraction=0.046, pad=0.03)
    ax.set_title(title)
    ax.set_xlabel("time [h]")
    return ax


def quantile_band(
    time_h: torch.Tensor,
    curves: torch.Tensor,
    color: str,
    label_text: str,
    ax: plt.Axes,
) -> None:
    """Median and 5-95 % / 25-75 % envelopes of a set of curves."""
    t, c = _np(time_h), _np(curves)
    q = np.quantile(c, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    ax.fill_between(t, q[0], q[4], color=color, alpha=0.15, lw=0)
    ax.fill_between(t, q[1], q[3], color=color, alpha=0.3, lw=0)
    ax.plot(t, q[2], color=color, lw=2, label=label_text)

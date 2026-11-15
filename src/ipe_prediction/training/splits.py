"""Reproducible, stratified train / validation / test split over all 1M simulations.

Why stratify? The EDA showed that the excitation received changes what the curve reveals
(never-on simulations are pure free responses, always-on ones are single steps, PRBS switching
ranges from minutes to hours). Every split must contain every kind of excitation *and* every kind
of building in the same proportions, so that validation and test scores measure the same thing
as training.

Strata are the cross product of:

* excitation regime: ``never_on`` / ``always_on`` / ``prbs``;
* for PRBS only: duty-cycle quintile x switching-speed (``t_min``) tercile;
* HLC quartile (building diversity).

Inside each stratum the simulations are shuffled with a seeded generator and cut 80/10/10. The
split is saved to disk the first time and re-loaded afterwards, so it is frozen across runs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ipe_prediction.data import TARGET_NAMES

logger = logging.getLogger(__name__)

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
REGIME_NAMES: tuple[str, ...] = ("never_on", "always_on", "prbs")


@dataclass
class ExcitationSummary:
    """Per-simulation descriptors used for stratification and reporting (all shape ``(N,)``)."""

    regime: torch.Tensor  # int64 index into REGIME_NAMES
    duty_cycle: torch.Tensor
    t_min_hours: torch.Tensor


def summarize_excitation(
    heater_on: torch.Tensor, t_min_hours: torch.Tensor, chunk: int = 100_000
) -> ExcitationSummary:
    """Compute regime and duty cycle from the binary heater series, chunk by chunk."""
    duty = torch.cat(
        [heater_on[i : i + chunk].float().mean(1) for i in range(0, heater_on.shape[0], chunk)]
    )
    regime = torch.full_like(duty, 2, dtype=torch.long)
    regime[duty == 0] = 0
    regime[duty == 1] = 1
    return ExcitationSummary(regime=regime, duty_cycle=duty, t_min_hours=t_min_hours.float())


def _quantile_bins(x: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Bin index in ``[0, n_bins)`` from the empirical quantiles of ``x``."""
    edges = torch.quantile(x.double(), torch.linspace(0, 1, n_bins + 1, dtype=torch.float64)[1:-1])
    return torch.bucketize(x.double(), edges)


def make_strata(
    exc: ExcitationSummary,
    hlc: torch.Tensor,
    n_duty_bins: int = 5,
    n_tmin_bins: int = 3,
    n_hlc_bins: int = 4,
) -> torch.Tensor:
    """Integer stratum label per simulation (see module docstring)."""
    prbs = exc.regime == 2
    duty_bin = torch.zeros_like(exc.regime)
    tmin_bin = torch.zeros_like(exc.regime)
    duty_bin[prbs] = _quantile_bins(exc.duty_cycle[prbs], n_duty_bins)
    tmin_bin[prbs] = _quantile_bins(exc.t_min_hours[prbs], n_tmin_bins)
    hlc_bin = _quantile_bins(hlc, n_hlc_bins)
    return ((exc.regime * 10 + duty_bin) * 10 + tmin_bin) * 10 + hlc_bin


def stratified_split(
    strata: torch.Tensor, fractions: tuple[float, float, float], seed: int
) -> dict[str, torch.Tensor]:
    """Shuffle each stratum with a seeded generator and cut it into train / val / test."""
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    _, f_val, f_test = fractions
    gen = torch.Generator().manual_seed(seed)
    parts: dict[str, list[torch.Tensor]] = {name: [] for name in SPLIT_NAMES}
    for s in torch.unique(strata):  # sorted -> deterministic order
        idx = torch.nonzero(strata == s).flatten()
        idx = idx[torch.randperm(len(idx), generator=gen)]
        n_test, n_val = round(len(idx) * f_test), round(len(idx) * f_val)
        parts["test"].append(idx[:n_test])
        parts["val"].append(idx[n_test : n_test + n_val])
        parts["train"].append(idx[n_test + n_val :])
    return {name: torch.cat(v).sort().values for name, v in parts.items()}


def save_split(
    path: Path, split: dict[str, torch.Tensor], seed: int, fractions: tuple[float, ...]
) -> None:
    """Persist the split indices (``.npz``) and a small JSON description."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: v.numpy().astype(np.int32) for k, v in split.items()})
    meta = {"seed": seed, "fractions": list(fractions), **{k: len(v) for k, v in split.items()}}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def load_split_indices(path: Path) -> dict[str, torch.Tensor]:
    """Load indices saved by :func:`save_split`."""
    with np.load(path) as f:
        return {name: torch.from_numpy(f[name].astype(np.int64)) for name in SPLIT_NAMES}


def get_or_create_split(
    path: Path,
    exc: ExcitationSummary,
    targets: torch.Tensor,
    fractions: tuple[float, float, float],
    seed: int,
) -> dict[str, torch.Tensor]:
    """Load the frozen split if it exists (checking seed and fractions), else create and save it."""
    if path.exists():
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta["seed"] != seed or not np.allclose(meta["fractions"], fractions):
            raise ValueError(f"{path} was created with {meta}; refusing to mix splits")
        logger.info("loaded frozen split %s", path)
        return load_split_indices(path)
    split = stratified_split(make_strata(exc, targets[:, 0]), fractions, seed)
    save_split(path, split, seed, fractions)
    logger.info("created split %s: %s", path, {k: len(v) for k, v in split.items()})
    return split


def representativeness_report(
    split: dict[str, torch.Tensor], exc: ExcitationSummary, targets: torch.Tensor
) -> pd.DataFrame:
    """Compare the splits: size, regime shares, excitation and target statistics."""
    rows = {}
    for name, idx in split.items():
        reg = exc.regime[idx]
        row: dict[str, float] = {"n": float(len(idx))}
        for r, rname in enumerate(REGIME_NAMES):
            row[f"share_{rname}_%"] = float((reg == r).float().mean() * 100)
        row["duty_cycle_mean"] = float(exc.duty_cycle[idx].mean())
        row["t_min_h_mean"] = float(exc.t_min_hours[idx].mean())
        y = targets[idx]
        for j, t in enumerate(TARGET_NAMES):
            row[f"{t}_mean"] = float(y[:, j].mean())
            row[f"{t}_std"] = float(y[:, j].std())
        rows[name] = row
    return pd.DataFrame(rows).T

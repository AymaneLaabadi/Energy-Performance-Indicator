"""Evaluation metrics, reported per target and per family of indicators, in physical units."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import torch

from ipe_prediction.data import TARGET_FAMILIES, TARGET_NAMES, TARGET_UNITS


def per_target_metrics(
    y_pred: torch.Tensor, y_true: torch.Tensor, names: Sequence[str] = TARGET_NAMES
) -> pd.DataFrame:
    """Regression metrics for each target (columns of ``y_pred`` / ``y_true``, named ``names``).

    Columns:
        ``unit``; ``MAE``, ``RMSE``, ``bias`` (mean of pred - true) and ``P95_AE`` (95th percentile of
        the absolute error) in physical units; ``MAPE_%``, ``P95_APE_%`` (relative errors); ``R2``.
    """
    yp, yt = y_pred.double().cpu(), y_true.double().cpu()
    err = yp - yt
    abs_err, rel_err = err.abs(), (err / yt).abs() * 100
    sst = (yt - yt.mean(0)).pow(2).sum(0)
    table = pd.DataFrame(
        {
            "MAE": abs_err.mean(0).numpy(),
            "RMSE": err.pow(2).mean(0).sqrt().numpy(),
            "bias": err.mean(0).numpy(),
            "P95_AE": abs_err.quantile(0.95, dim=0).numpy(),
            "MAPE_%": rel_err.mean(0).numpy(),
            "P95_APE_%": rel_err.quantile(0.95, dim=0).numpy(),
            "R2": (1 - err.pow(2).sum(0) / sst).numpy(),
        },
        index=list(names),
    )
    table.insert(0, "unit", [TARGET_UNITS[n] for n in names])
    return table


def per_family_metrics(table: pd.DataFrame) -> pd.DataFrame:
    """Average the relative metrics of :func:`per_target_metrics` within each family present."""
    rows = {
        family: table.loc[present, ["MAPE_%", "P95_APE_%", "R2"]].mean()
        for family, names in TARGET_FAMILIES.items()
        if (present := [n for n in names if n in table.index])
    }
    return pd.DataFrame(rows).T


def constraint_report(
    y_pred: torch.Tensor, names: Sequence[str] = TARGET_NAMES
) -> dict[str, float]:
    """Check the physical constraints that apply to the predicted targets."""
    y = dict(zip(names, y_pred.double().cpu().unbind(1), strict=True))
    report = {"share_all_positive": float((y_pred > 0).all(1).double().mean())}
    if {"HLC", "HLC_mit", "HLC_ext"} <= y.keys():
        gap = (y["HLC"] - y["HLC_mit"] - y["HLC_ext"]).abs().max()
        report["max_abs_HLC_minus_sum_W_per_K"] = float(gap)
    if {"tau_n", "tau_inf"} <= y.keys():
        report["share_tau_n_le_tau_inf"] = float((y["tau_n"] <= y["tau_inf"]).double().mean())
    if {"tau_inf", "tau_b"} <= y.keys():
        report["share_tau_inf_le_tau_b"] = float((y["tau_inf"] <= y["tau_b"]).double().mean())
    return report

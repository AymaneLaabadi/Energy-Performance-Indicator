"""Evaluation metrics, reported per target and per family of indicators, in physical units."""

from __future__ import annotations

import pandas as pd
import torch

from ipe_prediction.data import TARGET_FAMILIES, TARGET_NAMES, TARGET_UNITS


def per_target_metrics(y_pred: torch.Tensor, y_true: torch.Tensor) -> pd.DataFrame:
    """Regression metrics for each of the 6 targets.

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
        index=list(TARGET_NAMES),
    )
    table.insert(0, "unit", [TARGET_UNITS[n] for n in TARGET_NAMES])
    return table


def per_family_metrics(table: pd.DataFrame) -> pd.DataFrame:
    """Average the relative metrics of :func:`per_target_metrics` within each family."""
    rows = {
        family: table.loc[list(names), ["MAPE_%", "P95_APE_%", "R2"]].mean()
        for family, names in TARGET_FAMILIES.items()
    }
    return pd.DataFrame(rows).T


def constraint_report(y_pred: torch.Tensor) -> dict[str, float]:
    """Check the physical constraints on predictions (should hold by construction)."""
    hlc, mit, ext, tau_b, tau_n, tau_inf = y_pred.double().cpu().unbind(1)
    return {
        "max_abs_HLC_minus_sum_W_per_K": float((hlc - mit - ext).abs().max()),
        "share_tau_n_le_tau_inf": float((tau_n <= tau_inf).double().mean()),
        "share_tau_inf_le_tau_b": float((tau_inf <= tau_b).double().mean()),
        "share_all_positive": float((y_pred > 0).all(1).double().mean()),
    }

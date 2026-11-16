"""Data-quality checks on the raw batches and on the compact training tensors."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ipe_prediction.data import batch_files, targets_from_raw
from ipe_prediction.util_assessement import load_single_batch


def scan_raw_batches(folder: Path | str) -> pd.DataFrame:
    """One row per raw batch file: shapes, non-finite values, values of U, initial temperature,
    exact duplicate rows, and excitation settings that are constant within the file."""
    rows = []
    for path in batch_files(folder):
        raw = load_single_batch(path)
        t, u = raw.indoor_temperature, raw.indoor_U
        y = targets_from_raw(raw)
        u_values = torch.unique(u.round(decimals=1))
        both = torch.cat([t, u], dim=1)
        constant = [
            n for j, n in enumerate(raw.solicitation_field_names)
            if torch.unique(raw.solicitation[:, j]).numel() == 1
        ]  # fmt: skip
        rows.append({
            "file": path.name,
            "n": t.shape[0],
            "length": t.shape[1],
            "non-finite T": int((~torch.isfinite(t)).sum()),
            "non-finite U": int((~torch.isfinite(u)).sum()),
            "non-finite targets": int((~torch.isfinite(y)).sum()),
            "U values (W)": ", ".join(f"{v:g}" for v in u_values[:5].tolist()),
            "distinct T(0)": torch.unique(t[:, 0]).numel(),
            "T min": float(t.min()),
            "T max": float(t.max()),
            "duplicate (T, U) rows": t.shape[0] - torch.unique(both, dim=0).shape[0],
            "constant settings": ", ".join(constant),
        })  # fmt: skip
        del raw, t, u, both
    return pd.DataFrame(rows).set_index("file")


def robust_z(x: torch.Tensor) -> torch.Tensor:
    """``(x - median) / (1.4826 MAD)`` per column (``1.4826 MAD`` = std for normal data)."""
    med = x.median(0).values
    mad = (x - med).abs().median(0).values * 1.4826
    return (x - med) / mad.clamp_min(1e-12)


def curve_anomalies(
    delta_t: torch.Tensor, heater_on: torch.Tensor, chunk: int = 100_000
) -> pd.DataFrame:
    """Per-simulation curve statistics used to flag broken or implausible series.

    Columns: largest 1-min jump (°C), final rise, minimum rise, largest drop (°C) over a
    continuous heating period of at least one hour, number of distinct values.
    """
    out = []
    for d, u in zip(delta_t.split(chunk), heater_on.split(chunk), strict=True):
        u = u.float()
        jump = d.diff(dim=1).abs().max(1).values
        # drop over 60 min while the heater stayed on during those 60 min
        on_hour = u.unfold(1, 61, 1).min(2).values  # (n, L - 60) 1 if on the whole hour
        drop = (d[:, :-60] - d[:, 60:]) * on_hour
        distinct = (d.diff(dim=1) != 0).sum(1) + 1
        out.append(
            torch.stack([jump, d[:, -1], d.min(1).values, drop.max(1).values, distinct.float()], 1)
        )
    cols = [
        "max 1-min jump",
        "final rise",
        "min rise",
        "max 1-h drop while heating",
        "distinct values",
    ]
    return pd.DataFrame(torch.cat(out).numpy(), columns=cols)


def group_outliers(curves: torch.Tensor, group: torch.Tensor, min_size: int = 1000) -> torch.Tensor:
    """For each simulation in a group of identical inputs (``>= min_size``), the largest robust
    z-score of its curve against the other curves of the group, over time; NaN elsewhere."""
    score = torch.full((len(curves),), float("nan"))
    sizes = torch.bincount(group)
    for g in (sizes >= min_size).nonzero().flatten():
        idx = (group == g).nonzero().flatten()
        score[idx] = robust_z(curves[idx].double()).abs().max(1).values.float()
    return score


def contradictory_duplicates(
    delta_t: torch.Tensor, heater_on: torch.Tensor, targets: torch.Tensor
) -> dict:
    """Simulations with exactly the same input and the same (rounded) curve.

    Returns the number of such simulations and how many of them have different targets
    (identical observations with different labels: irreducible for any model).
    """
    keys = []  # 64-bit hash of (input, curve in 0.01 °C units) per simulation, memory-light
    for d, u in zip(delta_t.split(100_000), heater_on.split(100_000), strict=True):
        packed = np.packbits(u.numpy().astype(np.uint8), axis=1)
        centi = (d * 100).round().numpy().astype(np.int16)
        rows = np.concatenate([packed, centi.view(np.uint8)], axis=1)
        keys += [
            int.from_bytes(
                hashlib.blake2b(r.tobytes(), digest_size=8).digest(), "little", signed=True
            )
            for r in rows
        ]
    _, inv, counts = torch.unique(torch.tensor(keys), return_inverse=True, return_counts=True)
    dup = counts[inv] > 1
    n_dup = int(dup.sum())
    if n_dup == 0:
        return {"simulations with a duplicate observation": 0, "of which with different targets": 0}
    diff = 0
    for g in torch.unique(inv[dup]):
        rows = targets[inv == g]
        diff += int((rows - rows[0]).abs().max() > 0) * len(rows)
    return {
        "simulations with a duplicate observation": n_dup,
        "of which with different targets": diff,
    }

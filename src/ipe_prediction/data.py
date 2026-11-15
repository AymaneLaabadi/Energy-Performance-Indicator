"""Shared definitions for the ``quentin_house`` simulation dataset (used by training and EDA).

The raw batches are pickled ``ToSaveSimul`` objects (see :mod:`ipe_prediction.util_assessement`,
the loader provided with the assessment). This module fixes the canonical target order, units and
quantisation, and lists / decodes the raw batch files.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

from ipe_prediction.util_assessement import ToSaveSimul

#: Sampling period of the simulated series, in hours (1 minute; 480 steps = 8 h).
DT_HOURS: float = 1.0 / 60.0

#: Canonical target order used throughout the project.
TARGET_NAMES: tuple[str, ...] = ("HLC", "HLC_mit", "HLC_ext", "tau_b", "tau_n", "tau_inf")
TARGET_UNITS: dict[str, str] = {
    "HLC": "W/K",
    "HLC_mit": "W/K",
    "HLC_ext": "W/K",
    "tau_b": "h",
    "tau_n": "h",
    "tau_inf": "h",
}
TARGET_FAMILIES: dict[str, tuple[str, ...]] = {
    "HLC": ("HLC",),
    "Hb": ("HLC_mit", "HLC_ext"),
    "CDT": ("tau_b", "tau_n", "tau_inf"),
}

#: Resolution applied by the assessment's reference loading code (sensor-like quantisation).
TEMPERATURE_DECIMALS: int = 2
POWER_DECIMALS: int = 1


def batch_files(folder: Path | str) -> list[Path]:
    """List ``batch_*.pt`` files sorted by their numeric index (1, 2, ..., 20)."""
    files = list(Path(folder).glob("batch_*.pt"))
    if not files:
        raise FileNotFoundError(f"No batch_*.pt files found in {folder}")

    def _key(p: Path) -> int:
        match = re.search(r"batch_(\d+)\.pt$", p.name)
        return int(match.group(1)) if match else 10**9

    return sorted(files, key=_key)


def targets_from_raw(raw: ToSaveSimul) -> torch.Tensor:
    """Build the ``(N, 6)`` target matrix in canonical order from a raw batch."""
    hb_labels = list(raw.Hb_labels)
    hb = {name: raw.Hb[:, hb_labels.index(name)] for name in ("HLC_mit", "HLC_ext")}
    return torch.stack(
        [raw.HLC, hb["HLC_mit"], hb["HLC_ext"], raw.CDT[:, 0], raw.CDT[:, 1], raw.CDT[:, 2]],
        dim=1,
    ).float()

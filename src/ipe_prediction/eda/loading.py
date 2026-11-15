"""Raw-series loaders for the exploratory analysis (not used by training).

* :func:`load_dataset` loads time series + targets for a subset of batches;
* :func:`load_targets_only` scans *all* batches but keeps only the (tiny) target and
  excitation-parameter tensors, so the full 1M-simulation target distribution can be studied
  without holding ~4 GB of time series in memory.

Training uses the compact tensors of :mod:`ipe_prediction.training.dataset` instead.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from ipe_prediction.data import (
    DT_HOURS,
    POWER_DECIMALS,
    TARGET_NAMES,
    TEMPERATURE_DECIMALS,
    batch_files,
    targets_from_raw,
)
from ipe_prediction.util_assessement import load_single_batch


@dataclass
class SimulationSet:
    """A set of simulations with inputs, targets and excitation metadata.

    Attributes:
        temperature: Indoor air temperature ``Tint``, shape ``(N, L)`` in degC.
        power: Heating power ``U`` injected in the air node, shape ``(N, L)`` in W.
        targets: Target matrix, shape ``(N, 6)``, columns ordered as :data:`TARGET_NAMES`.
        excitation: Excitation-generator parameters, shape ``(N, P)``.
        excitation_names: Names of the ``P`` excitation parameters.
        batch_id: Index of the source batch file for every simulation, shape ``(N,)``.
    """

    temperature: torch.Tensor
    power: torch.Tensor
    targets: torch.Tensor
    excitation: torch.Tensor
    excitation_names: list[str]
    batch_id: torch.Tensor

    @property
    def n_sims(self) -> int:
        """Number of simulations."""
        return int(self.targets.shape[0])

    @property
    def n_steps(self) -> int:
        """Number of time steps per simulation (0 if series were not loaded)."""
        return int(self.temperature.shape[1]) if self.temperature.ndim == 2 else 0

    @property
    def time_hours(self) -> torch.Tensor:
        """Time axis in hours, shape ``(L,)``."""
        return torch.arange(self.n_steps, dtype=torch.float32) * DT_HOURS

    def target(self, name: str) -> torch.Tensor:
        """Return one target column by name."""
        return self.targets[:, TARGET_NAMES.index(name)]

    def subset(self, idx: torch.Tensor) -> SimulationSet:
        """Return a new :class:`SimulationSet` restricted to the rows ``idx``."""
        has_series = self.n_steps > 0
        return SimulationSet(
            temperature=self.temperature[idx] if has_series else self.temperature,
            power=self.power[idx] if has_series else self.power,
            targets=self.targets[idx],
            excitation=self.excitation[idx],
            excitation_names=self.excitation_names,
            batch_id=self.batch_id[idx],
        )


def load_dataset(
    folder: Path | str,
    batch_indices: Sequence[int] | None = None,
    quantize: bool = True,
) -> SimulationSet:
    """Load time series and targets for a subset of batches.

    Args:
        folder: Folder containing ``batch_*.pt``.
        batch_indices: 1-based batch numbers to load (e.g. ``[1, 2]``); ``None`` loads all.
        quantize: Round temperature to 0.01 degC and power to 0.1 W, as in the assessment's
            reference loading code.

    Returns:
        The concatenated :class:`SimulationSet`.
    """
    files = batch_files(folder)
    numbers = [int(re.search(r"(\d+)", f.stem).group(1)) for f in files]  # type: ignore[union-attr]
    wanted = set(batch_indices) if batch_indices is not None else set(numbers)
    temps, powers, targets, excit, ids = [], [], [], [], []
    names: list[str] = []
    for number, path in zip(numbers, files, strict=True):
        if number not in wanted:
            continue
        raw = load_single_batch(path)
        t, u = raw.indoor_temperature, raw.indoor_U
        if quantize:
            t = t.round(decimals=TEMPERATURE_DECIMALS)
            u = u.round(decimals=POWER_DECIMALS)
        temps.append(t.float())
        powers.append(u.float())
        targets.append(targets_from_raw(raw))
        excit.append(raw.solicitation.float())
        ids.append(torch.full((raw.HLC.shape[0],), number, dtype=torch.int16))
        names = list(raw.solicitation_field_names)
        del raw
    return SimulationSet(
        temperature=torch.cat(temps),
        power=torch.cat(powers),
        targets=torch.cat(targets),
        excitation=torch.cat(excit),
        excitation_names=names,
        batch_id=torch.cat(ids),
    )


def load_targets_only(folder: Path | str) -> SimulationSet:
    """Scan every batch but keep only targets and excitation parameters (low memory)."""
    targets, excit, ids = [], [], []
    names: list[str] = []
    for path in batch_files(folder):
        number = int(re.search(r"(\d+)", path.stem).group(1))  # type: ignore[union-attr]
        raw = load_single_batch(path)
        targets.append(targets_from_raw(raw))
        excit.append(raw.solicitation.float())
        ids.append(torch.full((raw.HLC.shape[0],), number, dtype=torch.int16))
        names = list(raw.solicitation_field_names)
        del raw
    return SimulationSet(
        temperature=torch.empty(0),
        power=torch.empty(0),
        targets=torch.cat(targets),
        excitation=torch.cat(excit),
        excitation_names=names,
        batch_id=torch.cat(ids),
    )

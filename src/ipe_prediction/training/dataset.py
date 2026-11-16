"""In-memory simulation tensors, PyTorch ``Dataset`` and ``DataLoader`` factory.

All 1M simulations are loaded once into compact tensors (~2.4 GB):

* ``delta_t``: ``Tint(t) - Tint(0)`` as float32 (every simulation starts at the same ``Tint(0)``,
  so the rise is the informative signal);
* ``heater_on``: ``U(t) > 0`` as uint8 (``U`` is binary: 0 or ``P_MAX`` W).

A split is then just a set of row indices (see :mod:`ipe_prediction.training.splits`): datasets index the
shared tensors instead of copying them.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ipe_prediction.data import (
    POWER_DECIMALS,
    TEMPERATURE_DECIMALS,
    batch_files,
    targets_from_raw,
)
from ipe_prediction.training.config import DataConfig
from ipe_prediction.training.splits import (
    get_or_create_split,
    representativeness_report,
    summarize_excitation,
)
from ipe_prediction.util_assessement import load_single_batch

logger = logging.getLogger(__name__)

#: Heater power when on (W). Checked in the EDA: ``U`` only takes the values 0 and 7000.
P_MAX: float = 7000.0


@dataclass
class SimulationTensors:
    """Compact in-memory storage of all simulations (row ``i`` = global simulation index ``i``).

    Attributes:
        delta_t: ``(N, L)`` float32 indoor temperature rise since t = 0 (K).
        heater_on: ``(N, L)`` uint8, 1 where the heater is on.
        targets: ``(N, 6)`` float32 targets in canonical order (``data.TARGET_NAMES``).
        excitation: ``(N, P)`` float32 excitation-generator settings (analysis only, never an input).
        excitation_names: names of the ``P`` settings.
        batch_id: ``(N,)`` int16 source batch file of each simulation.
    """

    delta_t: torch.Tensor
    heater_on: torch.Tensor
    targets: torch.Tensor
    excitation: torch.Tensor
    excitation_names: list[str]
    batch_id: torch.Tensor

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def excitation_column(self, name: str) -> torch.Tensor:
        """One excitation setting by name, e.g. ``"prbs_min_duration_hours"``."""
        return self.excitation[:, self.excitation_names.index(name)]


def load_all(folder: str, quantize: bool = True) -> SimulationTensors:
    """Load every ``batch_*.pt`` file into preallocated compact tensors.

    Args:
        folder: Folder containing the batch files.
        quantize: Apply the assessment's rounding (0.01 degC, 0.1 W) first.
    """
    files = batch_files(folder)
    out: SimulationTensors | None = None
    row = 0
    for path in files:
        raw = load_single_batch(path)
        temp, power = raw.indoor_temperature, raw.indoor_U
        if quantize:
            temp = temp.round(decimals=TEMPERATURE_DECIMALS)
            power = power.round(decimals=POWER_DECIMALS)
        n, length = temp.shape
        if out is None:  # preallocate once the per-batch shape is known
            total = n * len(files)
            out = SimulationTensors(
                delta_t=torch.empty(total, length, dtype=torch.float32),
                heater_on=torch.empty(total, length, dtype=torch.uint8),
                targets=torch.empty(total, 6, dtype=torch.float32),
                excitation=torch.empty(total, raw.solicitation.shape[1], dtype=torch.float32),
                excitation_names=list(raw.solicitation_field_names),
                batch_id=torch.empty(total, dtype=torch.int16),
            )
        sl = slice(row, row + n)
        out.delta_t[sl] = temp - temp[:, :1]
        out.heater_on[sl] = (power > 0.5 * P_MAX).to(torch.uint8)
        out.targets[sl] = targets_from_raw(raw)
        out.excitation[sl] = raw.solicitation.float()
        out.batch_id[sl] = int(path.stem.split("_")[1])
        row += n
        del raw, temp, power
    if out is None or row != len(out):
        raise RuntimeError(f"expected {len(files)} equally sized batches in {folder}")
    logger.info("loaded %d simulations from %d files", row, len(files))
    return out


def prepare_data(cfg: DataConfig) -> tuple[SimulationTensors, dict[str, torch.Tensor]]:
    """Load all simulations and the frozen stratified split (created on first use)."""
    tensors = load_all(cfg.folder, cfg.quantize)
    exc = summarize_excitation(
        tensors.heater_on, tensors.excitation_column("prbs_min_duration_hours")
    )
    split = get_or_create_split(
        cfg.split_path, exc, tensors.targets, cfg.split_fractions, cfg.split_seed
    )
    report_path = cfg.split_path.with_name(cfg.split_path.stem + "_report.csv")
    if not report_path.exists():
        representativeness_report(split, exc, tensors.targets).to_csv(report_path)
    return tensors, split


class ThermalDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset over a subset (``indices``) of :class:`SimulationTensors`.

    ``__getitems__`` lets the ``DataLoader`` gather a whole batch with one tensor indexing
    operation instead of one Python call per sample.

    Each batch is a dict with ``delta_t`` ``(B, L)`` in K, ``heater_on`` ``(B, L)`` in {0, 1},
    ``targets`` ``(B, 6)`` physical targets and ``index`` ``(B,)`` global simulation indices.
    """

    def __init__(
        self,
        tensors: SimulationTensors,
        indices: torch.Tensor,
        temperature_noise_std: float = 0.0,
    ) -> None:
        self.tensors = tensors
        self.indices = indices.long()
        self.temperature_noise_std = temperature_noise_std

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitems__(self, positions: Sequence[int]) -> dict[str, torch.Tensor]:
        idx = self.indices[torch.as_tensor(positions, dtype=torch.long)]
        delta_t = self.tensors.delta_t[idx]
        if self.temperature_noise_std > 0:
            delta_t = delta_t + self.temperature_noise_std * torch.randn_like(delta_t)
        return {
            "delta_t": delta_t,
            "heater_on": self.tensors.heater_on[idx].float(),
            "targets": self.tensors.targets[idx],
            "index": idx,
        }

    def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
        return {k: v[0] for k, v in self.__getitems__([position]).items()}


def _passthrough(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Collate function for datasets that already return whole batches."""
    return batch


def make_dataloader(
    dataset: ThermalDataset,
    batch_size: int,
    sampler: Sampler[int] | None = None,
) -> DataLoader[dict[str, torch.Tensor]]:
    """``DataLoader`` with batched fetching; in order unless a ``sampler`` (e.g. shuffling) is given."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=_passthrough,
        pin_memory=torch.cuda.is_available(),
    )

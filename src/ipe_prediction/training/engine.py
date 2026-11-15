"""Training / evaluation loops and checkpoint loading, shared by ``train.py`` and ``evaluate.py``.

``build_model`` is re-exported from :mod:`ipe_prediction.training.models`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ipe_prediction.training.config import ExperimentConfig
from ipe_prediction.training.losses import StandardizedMSELoss
from ipe_prediction.training.metrics import per_target_metrics
from ipe_prediction.training.models import ThermalRegressor, build_model
from ipe_prediction.training.preprocessing import TargetScaler

__all__ = ["autocast", "build_model", "evaluate", "load_run", "train_one_epoch"]


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def autocast(device: torch.device, amp: bool) -> torch.autocast:
    """bfloat16 autocast on CUDA when ``amp`` is set, a no-op otherwise."""
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: StandardizedMSELoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    amp: bool = False,
) -> float:
    """One pass over the training data; returns the mean training loss.

    The forward pass runs under bf16 autocast if ``amp``; the loss is computed in float32. The loss
    is accumulated on the device so there is a single host sync per epoch.
    """
    model.train()
    total = torch.zeros((), device=device)
    n = 0
    for batch in loader:
        batch = _to_device(batch, device)
        with autocast(device, amp):
            out = model(batch["delta_t"], batch["heater_on"])
        loss, _ = loss_fn(out.float(), batch["targets"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        bs = batch["targets"].shape[0]
        total += loss.detach() * bs
        n += bs
    return float(total) / max(n, 1)


@torch.inference_mode()
def evaluate(
    model: ThermalRegressor,
    loader: DataLoader,
    loss_fn: StandardizedMSELoss,
    device: torch.device,
    amp: bool = False,
) -> dict[str, Any]:
    """Loss, per-target loss terms, physical predictions and metrics on one split."""
    model.eval()
    outputs, targets, indices = [], [], []
    for batch in loader:
        batch = _to_device(batch, device)
        with autocast(device, amp):
            outputs.append(model(batch["delta_t"], batch["heater_on"]).float())
        targets.append(batch["targets"])
        indices.append(batch["index"])
    out, y = torch.cat(outputs), torch.cat(targets)
    loss, per_target = loss_fn(out, y)
    pred = model.target_scaler.decode(out).cpu()
    return {
        "loss": loss.item(),
        "loss_per_target": per_target.cpu(),
        "pred": pred,
        "target": y.cpu(),
        "index": torch.cat(indices).cpu(),
        "metrics": per_target_metrics(pred, y.cpu()),
    }


def load_run(run_dir: Path | str, device: str = "cpu") -> tuple[ThermalRegressor, ExperimentConfig]:
    """Reload the best model of a run (scaling statistics are restored from the buffers)."""
    ckpt = torch.load(Path(run_dir) / "best.pt", map_location=device, weights_only=False)
    cfg = ExperimentConfig.from_dict(ckpt["config"])
    model = build_model(cfg.model, 1.0, TargetScaler.empty())
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), cfg

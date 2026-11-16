"""Training and evaluation loops, learning-rate schedule, mixed precision, checkpoint loading."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ipe_prediction.training.config import ExperimentConfig
from ipe_prediction.training.losses import standardized_mse
from ipe_prediction.training.metrics import per_target_metrics
from ipe_prediction.training.models import build_model
from ipe_prediction.training.preprocessing import Preprocessor, n_inputs


def amp_dtype(device: torch.device, enabled: bool) -> torch.dtype | None:
    """Autocast dtype: bf16 on GPUs with native support (compute capability >= 8), fp16 otherwise
    (T4, P100); None (float32) when disabled or on CPU."""
    if not enabled or device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


def warmup_cosine(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int, min_ratio: float
) -> torch.optim.lr_scheduler.LambdaLR:
    """Per-step schedule: linear warmup to the base lr, then cosine decay to ``min_ratio x lr``."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _forward(
    model: nn.Module,
    prep: Preprocessor,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    delta_t = batch["delta_t"].to(device, non_blocking=True)
    heater_on = batch["heater_on"].to(device, non_blocking=True)
    with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
        return model(prep.inputs(delta_t, heater_on)).float()


def train_one_epoch(
    model: nn.Module,
    prep: Preprocessor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    grad_scaler: torch.amp.GradScaler,
    device: torch.device,
    dtype: torch.dtype | None,
    grad_clip: float,
) -> float:
    """One pass over the training data; returns the mean training loss.

    ``grad_scaler`` is only active for fp16 (it is a pass-through when disabled).
    """
    model.train()
    total, n = torch.zeros((), device=device), 0
    for batch in loader:
        targets = batch["targets"].to(device, non_blocking=True)
        loss, _ = standardized_mse(_forward(model, prep, batch, device, dtype), targets, prep)
        optimizer.zero_grad(set_to_none=True)
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        scheduler.step()
        total += loss.detach() * len(targets)
        n += len(targets)
    return float(total) / max(n, 1)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    prep: Preprocessor,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Loss, per-target loss, physical predictions, targets, indices and metrics on one split."""
    model.eval()
    outputs, targets, indices = [], [], []
    for batch in loader:
        outputs.append(_forward(model, prep, batch, device, dtype))
        targets.append(batch["targets"].to(device))
        indices.append(batch["index"])
    out, y = torch.cat(outputs), torch.cat(targets)
    loss, per_target = standardized_mse(out, y, prep)
    pred = prep.decode(out).cpu()
    return {
        "loss": loss.item(),
        "loss_per_target": per_target.cpu(),
        "pred": pred,
        "target": y.cpu(),
        "index": torch.cat(indices),
        "metrics": per_target_metrics(pred, y.cpu()),
    }


def load_model(
    run_dir: Path | str, model: nn.Module | None = None, device: str = "cpu"
) -> tuple[nn.Module, Preprocessor, ExperimentConfig]:
    """Best checkpoint of a run. Pass ``model`` (untrained) for a model outside the registry."""
    cfg = ExperimentConfig.load(run_dir)
    ckpt = torch.load(Path(run_dir) / "best.pt", map_location=device, weights_only=True)
    if model is None:
        model = build_model(cfg.model, cfg.model_config, n_inputs(cfg.data.extra_features))
    model.load_state_dict(ckpt["model"])
    prep = Preprocessor(cfg.data.extra_features)
    prep.load_state_dict(ckpt["preprocessor"])
    return model.to(device).eval(), prep.to(device), cfg

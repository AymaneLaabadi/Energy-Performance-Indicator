"""Training and evaluation loops, learning-rate schedule, mixed precision, checkpoint loading."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ipe_prediction.training.config import ExperimentConfig
from ipe_prediction.training.losses import n_network_outputs, target_loss
from ipe_prediction.training.metrics import per_target_metrics
from ipe_prediction.training.models import build_model
from ipe_prediction.training.preprocessing import Preprocessor, n_inputs


def amp_dtype(device: torch.device, enabled: bool) -> torch.dtype | None:
    """Autocast dtype: bf16 on GPUs with native support (compute capability >= 8), fp16 otherwise
    (T4, P100); None (float32) when disabled or on CPU."""
    if not enabled or device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


def lr_scheduler(
    optimizer: torch.optim.Optimizer,
    schedule: str,
    total_steps: int,
    warmup_steps: int,
    min_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Per-step schedule, as a factor of the optimizer's base lr (see ``TrainConfig``).

    * ``"step"``: 1 for the first ``warmup_steps``, then ``min_ratio`` until the end;
    * ``"cosine"``: linear ramp to 1 over ``warmup_steps``, then cosine decay to ``min_ratio``.
    """
    if schedule not in ("step", "cosine"):
        raise ValueError(f"lr_schedule must be 'step' or 'cosine', got {schedule!r}")

    def factor(step: int) -> float:
        if schedule == "step":
            return 1.0 if step < warmup_steps else min_ratio
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


@torch.no_grad()
def recalibrate_batchnorm(
    model: nn.Module,
    prep: Preprocessor,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype | None,
    n_batches: int,
) -> None:
    """Recompute the BatchNorm running statistics exactly ("precise BN"), weights frozen.

    During training the running statistics are a moving average over the last batches. The tests
    mix very different curves (a rise of ~1 °C without heating, ~15 °C always heated), so these
    statistics depend on the last batches seen and the evaluated model jumps from epoch to epoch
    even when the weights barely change. Here they are replaced by the plain average over
    ``n_batches`` training batches, so that the evaluation reflects the weights only.
    """
    norms = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not norms or n_batches <= 0:
        return
    momenta = [m.momentum for m in norms]
    for m in norms:
        m.reset_running_stats()
        m.momentum = None  # cumulative average
    model.train()
    for i, batch in enumerate(loader):
        if i == n_batches:
            break
        _forward(model, prep, batch, device, dtype)
    for m, momentum in zip(norms, momenta, strict=True):
        m.momentum = momentum
    model.eval()


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
    loss_kind: str = "mse",
) -> float:
    """One pass over the training data; returns the mean training objective (see losses).

    ``grad_scaler`` is only active for fp16 (it is a pass-through when disabled).
    """
    model.train()
    total, n = torch.zeros((), device=device), 0
    for batch in loader:
        targets = batch["targets"].to(device, non_blocking=True)
        out = _forward(model, prep, batch, device, dtype)
        loss, _ = target_loss(out, targets, prep, loss_kind)
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
    loss_kind: str = "mse",
) -> dict[str, Any]:
    """Loss, per-target loss, physical predictions and targets (the selected targets, see
    ``prep.target_names``), indices and metrics on one split. ``loss`` is always the
    standardised MSE of the means; with the Gaussian NLL, ``pred_std_z`` holds the predicted
    standard deviations in the standardised space ``(N, T)``."""
    model.eval()
    outputs, targets, indices = [], [], []
    for batch in loader:
        outputs.append(_forward(model, prep, batch, device, dtype))
        targets.append(batch["targets"].to(device))
        indices.append(batch["index"])
    out, y = torch.cat(outputs), torch.cat(targets)
    loss, per_target = target_loss(out, y, prep, "mse")
    mean = out[:, : prep.n_outputs]
    pred, target = prep.decode(mean).cpu(), prep.select(y).cpu()
    extra = {}
    if loss_kind == "gaussian_nll":
        var = prep.standardized_variance(out[:, prep.n_outputs :].clamp(-10, 10).exp())
        extra["pred_std_z"] = var.sqrt().cpu()
    return extra | {
        "loss": loss.item(),
        "loss_per_target": per_target.cpu(),
        "pred": pred,
        "target": target,
        "index": torch.cat(indices),
        "metrics": per_target_metrics(pred, target, prep.target_names),
    }


def load_model(
    run_dir: Path | str, model: nn.Module | None = None, device: str = "cpu"
) -> tuple[nn.Module, Preprocessor, ExperimentConfig]:
    """Best checkpoint of a run. Pass ``model`` (untrained) for a model outside the registry."""
    cfg = ExperimentConfig.load(run_dir)
    ckpt = torch.load(Path(run_dir) / "best.pt", map_location=device, weights_only=True)
    prep = Preprocessor(cfg.data.extra_features, cfg.data.targets)
    prep.load_state_dict(ckpt["preprocessor"])
    if model is None:
        in_channels = n_inputs(cfg.data.extra_features)
        n_out = n_network_outputs(prep, cfg.train.loss)
        model = build_model(cfg.model, cfg.model_config, in_channels, n_out)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), prep.to(device), cfg

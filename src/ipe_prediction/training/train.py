"""Training entry point (train + validation only; the test split is never loaded here).

Usage (from the project root)::

    uv run ipe-train                                   # defaults of config.py (TCN + attention)
    uv run ipe-train --name gru --arch cnn_gru         # the CNN-GRU baseline
    uv run ipe-train --name smoke --epochs 1 --max-train 50000
    uv run ipe-train --name smoke --resume             # continue a stopped run from its last epoch
    uv run ipe-train --name smoke --resume --epochs 80 # ... and extend it to 80 epochs

The run trains for a fixed number of epochs (no early stopping). Artefacts in
``outputs/runs/<name>/``:

* ``config.json``, ``train.log``, ``history.csv`` (one row per epoch);
* ``best.pt``: weights + config of the best validation epoch (used by ``ipe-evaluate``);
* ``last.pt``: full training state after the last finished epoch (model, optimizer, RNG), so that
  a run stopped at any point (Ctrl+C, crash) resumes exactly where it was;
* ``metrics_val.csv``, ``summary.json``: validation metrics of ``best.pt``, written at the end.

The test split is evaluated once, at the end of the project, with ``ipe-evaluate``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ipe_prediction.data import TARGET_NAMES
from ipe_prediction.training.config import (
    ARCHITECTURES,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
)
from ipe_prediction.training.dataset import (
    SimulationTensors,
    ThermalDataset,
    make_dataloader,
    prepare_data,
)
from ipe_prediction.training.engine import evaluate, train_one_epoch
from ipe_prediction.training.losses import StandardizedMSELoss
from ipe_prediction.training.metrics import constraint_report, per_family_metrics
from ipe_prediction.training.models import (
    CustomRegressor,
    ThermalRegressor,
    build_model,
    count_parameters,
    n_input_channels,
)
from ipe_prediction.training.preprocessing import TargetScaler, fit_delta_t_std

logger = logging.getLogger("ipe_prediction.training.train")

LAST, BEST = "last.pt", "best.pt"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _subsample(idx: torch.Tensor, n: int | None, seed: int) -> torch.Tensor:
    """Deterministic random subset of ``n`` indices (all of them if ``n`` is None)."""
    if n is None or n >= len(idx):
        return idx
    gen = torch.Generator().manual_seed(seed)
    return idx[torch.randperm(len(idx), generator=gen)[:n]].sort().values


def _atomic_save(obj: Any, path: Path) -> None:
    """``torch.save`` through a temporary file, so an interruption never leaves a corrupt file."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state()}
    state["torch"] = torch.get_rng_state()
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _make_model(
    cfg: ExperimentConfig,
    delta_t_std: float,
    scaler: TargetScaler,
    net: Callable[[], torch.nn.Module] | None,
    length: int,
) -> ThermalRegressor:
    """Built-in architecture from the config, or ``net()`` wrapped in a :class:`CustomRegressor`."""
    if net is None:
        return build_model(cfg.model, delta_t_std, scaler)
    model = CustomRegressor(net(), delta_t_std, scaler, cfg.model.tcn_extra_features)
    n_in = n_input_channels(cfg.model.tcn_extra_features)
    with torch.no_grad():  # fail early, with a clear message, on a wrong input/output contract
        out = model.eval()(torch.zeros(2, length), torch.zeros(2, length))
    if out.shape != (2, 5):
        raise ValueError(f"the network must map (B, {n_in}, L) to (B, 5); got {tuple(out.shape)}")
    return model.train()


def run(
    cfg: ExperimentConfig,
    resume: bool = False,
    net: Callable[[], torch.nn.Module] | None = None,
    data: tuple[SimulationTensors, dict[str, torch.Tensor]] | None = None,
) -> dict[str, object]:
    """Train for ``cfg.train.epochs`` epochs (resuming from ``last.pt`` if asked).

    Args:
        cfg: Experiment configuration (``cfg.model.arch == "custom"`` when ``net`` is given).
        resume: Continue from ``last.pt`` of ``cfg.run_dir``.
        net: Factory of a custom network ``(B, F, L) -> (B, 5)`` (see :class:`CustomRegressor`);
            ``None`` builds ``cfg.model.arch``.
        data: Already loaded ``prepare_data(cfg.data)`` output (avoids reloading in a notebook).

    Returns:
        The summary written to ``summary.json``, plus ``"model"``: the best model.
    """
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.json")
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    tensors, split = data if data is not None else prepare_data(cfg.data)
    # the test indices are never used here; both subsets are seeded, so a resumed run sees the
    # exact same training and validation simulations
    train_idx = _subsample(split["train"], cfg.data.max_train, cfg.train.seed)
    val_idx = _subsample(split["val"], cfg.data.val_subset, cfg.train.seed + 1)
    logger.info(
        "data ready in %.0fs | train %d | val %d", time.time() - t0, len(train_idx), len(val_idx)
    )

    scaler = TargetScaler.fit(tensors.targets[train_idx])
    delta_t_std = fit_delta_t_std(tensors.delta_t, train_idx)
    train_loader = make_dataloader(
        ThermalDataset(tensors, train_idx, cfg.data.temperature_noise_std),
        cfg.train.batch_size,
        shuffle=True,
        seed=cfg.train.seed,
    )
    shuffle_gen = train_loader.generator  # shared with its RandomSampler
    assert shuffle_gen is not None
    val_loader = make_dataloader(
        ThermalDataset(tensors, val_idx), cfg.train.eval_batch_size, shuffle=False
    )

    model = _make_model(cfg, delta_t_std, scaler, net, tensors.delta_t.shape[1]).to(device)
    loss_fn = StandardizedMSELoss(model.target_scaler)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    logger.info(
        "model %s: %s parameters | delta_t std %.3f K | batch %d | amp %s",
        cfg.model.arch,
        f"{count_parameters(model):,}",
        delta_t_std,
        cfg.train.batch_size,
        cfg.train.amp,
    )

    history: list[dict[str, float]] = []
    best_val, best_epoch, start_epoch = math.inf, -1, 1
    if resume:
        ckpt = torch.load(run_dir / LAST, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        _set_rng_state(ckpt["rng"])
        best_val, best_epoch, start_epoch = ckpt["best_val"], ckpt["best_epoch"], ckpt["epoch"] + 1
        if (run_dir / "history.csv").exists():
            hist = pd.read_csv(run_dir / "history.csv")
            history = hist[hist["epoch"] < start_epoch].to_dict(orient="records")
        logger.info(
            "resumed from epoch %d (best %d, val %.4f)", ckpt["epoch"], best_epoch, best_val
        )
    if start_epoch > cfg.train.epochs:
        logger.info("nothing to train: %d epochs already done", start_epoch - 1)

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        t_ep = time.time()
        # one shuffle per epoch derived from (seed, epoch): resuming gives the same order
        shuffle_gen.manual_seed(cfg.train.seed * 100_003 + epoch)
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, cfg.train.grad_clip, cfg.train.amp
        )
        val = evaluate(model, val_loader, loss_fn, device, cfg.train.amp)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "seconds": time.time() - t_ep,
            "lr": optimizer.param_groups[0]["lr"],
        }
        row |= {
            f"val_loss_{n}": float(v)
            for n, v in zip(TARGET_NAMES, val["loss_per_target"], strict=True)
        }
        row |= {f"val_MAPE_{n}": float(val["metrics"].loc[n, "MAPE_%"]) for n in TARGET_NAMES}
        history.append(row)
        logger.info(
            "epoch %03d | train %.4f | val %.4f | val MAPE %% %s | %.0fs | lr %.0e",
            epoch,
            train_loss,
            val["loss"],
            " ".join(f"{n}={row[f'val_MAPE_{n}']:.2f}" for n in TARGET_NAMES),
            row["seconds"],
            row["lr"],
        )
        if val["loss"] < best_val:
            best_val, best_epoch = val["loss"], epoch
            _atomic_save(
                {"model_state": model.state_dict(), "config": cfg.to_dict(), "epoch": epoch},
                run_dir / BEST,
            )
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        _atomic_save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": cfg.to_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "rng": _rng_state(),
            },
            run_dir / LAST,
        )

    model.load_state_dict(torch.load(run_dir / BEST, weights_only=False)["model_state"])
    val = evaluate(model, val_loader, loss_fn, device, cfg.train.amp)
    val["metrics"].to_csv(run_dir / "metrics_val.csv")
    summary = {
        "best_epoch": best_epoch,
        "epochs_done": max(start_epoch - 1, len(history)),
        "val_loss": val["loss"],
        "val_family": per_family_metrics(val["metrics"]).round(4).to_dict(orient="index"),
        "val_constraints": constraint_report(val["pred"]),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        "validation metrics (best epoch %d):\n%s", best_epoch, val["metrics"].round(3).to_string()
    )
    return summary | {"model": model}


def parse_args(argv: Sequence[str] | None = None) -> tuple[ExperimentConfig, bool]:
    """Build the run configuration: defaults of config.py, or the saved one with ``--resume``."""
    t, d, m = TrainConfig(), DataConfig(), ModelConfig()
    p = argparse.ArgumentParser(description="Train a model on the train split (validation only).")
    p.add_argument("--name", default=ExperimentConfig.name)
    p.add_argument("--arch", choices=ARCHITECTURES, help=f"model (default {m.arch})")
    p.add_argument("--resume", action="store_true", help="continue run <name> from its last epoch")
    p.add_argument("--epochs", type=int, help=f"total epochs (default {t.epochs})")
    p.add_argument("--batch-size", type=int, help=f"default {t.batch_size}")
    p.add_argument("--lr", type=float, help=f"default {t.lr}")
    p.add_argument("--weight-decay", type=float, help=f"default {t.weight_decay}")
    p.add_argument("--seed", type=int, help=f"default {t.seed}")
    p.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, help=f"bf16 autocast (default {t.amp})"
    )
    p.add_argument("--max-train", type=int, help="subsample the training split (quick runs)")
    a = p.parse_args(argv)

    run_dir = Path(ExperimentConfig.output_dir) / a.name
    overrides = {
        k: v
        for k, v in {
            "batch_size": a.batch_size,
            "lr": a.lr,
            "weight_decay": a.weight_decay,
            "seed": a.seed,
            "amp": a.amp,
            "max_train": a.max_train,
            "arch": a.arch,
        }.items()
        if v is not None
    }
    if a.resume:
        if not (run_dir / LAST).exists():
            p.error(f"--resume: no {LAST} in {run_dir}")
        if overrides:
            p.error(f"--resume continues the saved config; cannot change {sorted(overrides)}")
        cfg = ExperimentConfig.from_dict(json.loads((run_dir / "config.json").read_text()))
        if a.epochs is not None:
            cfg.train.epochs = a.epochs
        return cfg, True

    if (run_dir / LAST).exists():
        p.error(f"{run_dir} already holds a run: use --resume, or another --name")
    max_train = overrides.pop("max_train", d.max_train)
    arch = overrides.pop("arch", m.arch)
    if a.epochs is not None:
        overrides["epochs"] = a.epochs
    cfg = ExperimentConfig(
        name=a.name,
        data=DataConfig(max_train=max_train),
        model=ModelConfig(arch=arch),
        train=TrainConfig(**overrides),
    )
    return cfg, False


def setup_logging(log_file: Path, append: bool) -> None:
    """Log to the console and to ``log_file`` (appended to when resuming)."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    handler = logging.FileHandler(log_file, mode="a" if append else "w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point (``uv run ipe-train``)."""
    cfg, resume = parse_args(argv)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(cfg.run_dir / "train.log", append=resume)
    run(cfg, resume)


if __name__ == "__main__":
    main()

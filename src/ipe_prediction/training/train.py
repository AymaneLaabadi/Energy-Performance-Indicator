"""Train a model on the training split, scoring the validation split each epoch.

Usage (from the project root)::

    uv run ipe-train --name tcn                       # TCN + attention, default settings
    uv run ipe-train --name gru --model cnn_gru --epochs 30
    uv run ipe-train --name quick --epochs 3 --max-train 50000
    uv run ipe-train --name tcn --resume              # continue a stopped run

Artefacts in ``outputs/runs/<name>/``: ``config.json``, ``model.txt``, ``train.log``,
``history.csv`` (one row per epoch), ``best.pt`` (best validation epoch), ``last.pt`` (state to
resume), and at the end ``metrics_val.csv`` and ``summary.json`` for ``best.pt``.
The test split is never used here (see ``ipe-evaluate``).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from ipe_prediction.data import TARGET_NAMES
from ipe_prediction.training.config import DataConfig, ExperimentConfig, TrainConfig
from ipe_prediction.training.dataset import (
    SimulationTensors,
    ThermalDataset,
    make_dataloader,
    prepare_data,
)
from ipe_prediction.training.engine import amp_dtype, evaluate, train_one_epoch, warmup_cosine
from ipe_prediction.training.metrics import constraint_report, per_family_metrics
from ipe_prediction.training.models import MODELS, build_model, count_parameters
from ipe_prediction.training.preprocessing import Preprocessor, n_inputs

logger = logging.getLogger("ipe_prediction.train")

Data = tuple[SimulationTensors, dict[str, torch.Tensor]]


def subset(indices: torch.Tensor, n: int | None, seed: int) -> torch.Tensor:
    """Seeded random subset of ``n`` indices (all of them if ``n`` is None)."""
    if n is None or n >= len(indices):
        return indices
    perm = torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed))
    return indices[perm[:n]].sort().values


def run(
    cfg: ExperimentConfig,
    model: nn.Module | None = None,
    resume: bool = False,
    data: Data | None = None,
) -> dict:
    """Train ``cfg`` and return the validation summary of its best epoch.

    Args:
        cfg: Experiment configuration.
        model: An untrained network ``(B, C_in, L) -> (B, 5)`` defined outside the registry
            (e.g. in a notebook); None builds ``cfg.model`` from ``cfg.model_config``.
        resume: Continue from ``last.pt`` (same configuration).
        data: ``prepare_data(cfg.data)`` output, to avoid reloading it in a notebook.
    """
    run_dir = cfg.run_dir
    if (run_dir / "last.pt").exists() and not resume:
        raise FileExistsError(f"{run_dir} already holds a run: resume it or use another name")
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log_file = logging.FileHandler(run_dir / "train.log", mode="a" if resume else "w")
    log_file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(log_file)
    try:
        return _train(cfg, model, resume, data)
    finally:
        logging.getLogger().removeHandler(log_file)
        log_file.close()


def _train(cfg: ExperimentConfig, model: nn.Module | None, resume: bool, data: Data | None) -> dict:
    tc, run_dir = cfg.train, cfg.run_dir
    torch.manual_seed(tc.seed)
    device = torch.device(tc.device if torch.cuda.is_available() else "cpu")

    # --- data (seeded subsets: a resumed run sees the same simulations)
    tensors, split = data if data is not None else prepare_data(cfg.data)
    train_idx = subset(split["train"], cfg.data.max_train, tc.seed)
    val_idx = subset(split["val"], cfg.data.val_subset, tc.seed + 1)
    prep = Preprocessor.fit(tensors.delta_t, tensors.targets, train_idx, cfg.data.extra_features)
    prep = prep.to(device)
    train_set = ThermalDataset(tensors, train_idx, cfg.data.temperature_noise_std)
    train_loader = make_dataloader(train_set, tc.batch_size, shuffle=True, seed=tc.seed)
    val_loader = make_dataloader(ThermalDataset(tensors, val_idx), tc.eval_batch_size)

    # --- model and optimisation
    if model is None:
        model = build_model(cfg.model, cfg.model_config, n_inputs(cfg.data.extra_features))
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    steps = len(train_loader)
    scheduler = warmup_cosine(
        optimizer, tc.epochs * steps, round(tc.warmup_epochs * steps), tc.min_lr / tc.lr
    )
    dtype = amp_dtype(device, tc.amp)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

    start_epoch, best_val, best_epoch, history = 1, math.inf, 0, []
    if resume:
        ckpt = torch.load(run_dir / "last.pt", map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        grad_scaler.load_state_dict(ckpt["grad_scaler"])
        start_epoch, best_val, best_epoch = ckpt["epoch"] + 1, ckpt["best_val"], ckpt["best_epoch"]
        history = pd.read_csv(run_dir / "history.csv").to_dict("records")[: ckpt["epoch"]]
        logger.info(
            "resumed after epoch %d (best %d, val %.4f)", ckpt["epoch"], best_epoch, best_val
        )
    else:
        cfg.save(run_dir / "config.json")
        (run_dir / "model.txt").write_text(str(model))

    logger.info(
        "%s: %s parameters | train %d | val %d | batch %d | %s | lr %.1e, warmup %.1f epochs",
        cfg.model,
        f"{count_parameters(model):,}",
        len(train_idx),
        len(val_idx),
        tc.batch_size,
        str(dtype).replace("torch.", "autocast ") if dtype else "float32",
        tc.lr,
        tc.warmup_epochs,
    )

    # --- epochs
    for epoch in range(start_epoch, tc.epochs + 1):
        t0 = time.time()
        train_loader.generator.manual_seed(tc.seed * 100_003 + epoch)  # same order when resuming
        train_loss = train_one_epoch(
            model, prep, train_loader, optimizer, scheduler, grad_scaler, device, dtype,
            tc.grad_clip,
        )  # fmt: skip
        val = evaluate(model, prep, val_loader, device, dtype)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "seconds": time.time() - t0,
            "lr": optimizer.param_groups[0]["lr"],
            **{
                f"val_loss_{n}": float(v)
                for n, v in zip(TARGET_NAMES, val["loss_per_target"], strict=True)
            },
            **{f"val_MAPE_{n}": float(val["metrics"].loc[n, "MAPE_%"]) for n in TARGET_NAMES},
        }
        history.append(row)
        logger.info(
            "epoch %03d | train %.4f | val %.4f | val MAPE %% %s | %.0fs | lr %.1e",
            epoch,
            train_loss,
            val["loss"],
            " ".join(f"{n}={row[f'val_MAPE_{n}']:.2f}" for n in TARGET_NAMES),
            row["seconds"],
            row["lr"],
        )

        if val["loss"] < best_val:
            best_val, best_epoch = val["loss"], epoch
            torch.save(
                {"model": model.state_dict(), "preprocessor": prep.state_dict(), "epoch": epoch},
                run_dir / "best.pt",
            )
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "grad_scaler": grad_scaler.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "best_epoch": best_epoch,
            },
            run_dir / "last.pt",
        )

    # --- final validation of the best epoch
    model.load_state_dict(torch.load(run_dir / "best.pt", weights_only=True)["model"])
    val = evaluate(model, prep, val_loader, device, dtype)
    val["metrics"].to_csv(run_dir / "metrics_val.csv")
    summary = {
        "best_epoch": best_epoch,
        "val_loss": val["loss"],
        "val_family": per_family_metrics(val["metrics"]).round(4).to_dict(orient="index"),
        "val_constraints": constraint_report(val["pred"]),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        "validation metrics (best epoch %d):\n%s", best_epoch, val["metrics"].round(3).to_string()
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> tuple[ExperimentConfig, bool]:
    """Configuration from the command line (or the saved one with ``--resume``)."""
    t = TrainConfig()
    p = argparse.ArgumentParser(description="Train a model (train / validation splits only).")
    p.add_argument("--name", required=True, help="run name: outputs/runs/<name>")
    p.add_argument("--model", default="tcn_attn", choices=sorted(MODELS))
    p.add_argument("--resume", action="store_true", help="continue run <name> from last.pt")
    p.add_argument("--epochs", type=int, default=t.epochs)
    p.add_argument("--batch-size", type=int, default=t.batch_size)
    p.add_argument("--lr", type=float, default=t.lr)
    p.add_argument("--max-train", type=int, help="subsample the training split (quick runs)")
    p.add_argument("--extra-features", action="store_true", help="5 input channels instead of 2")
    p.add_argument("--no-amp", dest="amp", action="store_false", help="train in float32")
    a = p.parse_args(argv)

    if a.resume:
        return ExperimentConfig.load(Path(ExperimentConfig.output_dir) / a.name), True
    cfg = ExperimentConfig(
        name=a.name,
        model=a.model,
        data=DataConfig(max_train=a.max_train, extra_features=a.extra_features),
        train=TrainConfig(epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, amp=a.amp),
    )
    return cfg, False


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point (``uv run ipe-train``)."""
    cfg, resume = parse_args(argv)
    run(cfg, resume=resume)


if __name__ == "__main__":
    main()

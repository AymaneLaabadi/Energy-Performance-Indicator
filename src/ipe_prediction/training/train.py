"""Train a model on the training split, scoring the validation split each epoch.

Usage (from the project root)::

    uv run ipe-train --name tcn                       # TCN + attention, default settings
    uv run ipe-train --name gru --model cnn_gru --epochs 30
    uv run ipe-train --name quick --epochs 3 --max-train 50000
    uv run ipe-train --name tcn --resume              # continue a stopped run
    uv run ipe-train --name tcn --resume --epochs 900 # ... and extend it (constant / restarts)
    python -m ipe_prediction.training.train --name tcn --ddp   # all GPUs (Kaggle T4 x2)

Artefacts in ``outputs/runs/<name>/``: ``config.json``, ``model.txt``, ``train.log``,
``history.csv`` (one row per epoch), ``best.pt`` (best validation epoch), ``last.pt`` (state to
resume), and at the end ``metrics_val.csv`` and ``summary.json`` for ``best.pt``.
The test split is never used here (see ``ipe-evaluate``).

``--ddp`` starts one process per visible GPU (DistributedDataParallel). ``batch_size`` stays the
global batch (split across the GPUs), so a run gives the same kind of result with or without it
and can be resumed either way. Every process scores the validation set; process 0 logs and saves.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import logging
import math
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

from ipe_prediction.data import TARGET_NAMES
from ipe_prediction.training.config import DataConfig, ExperimentConfig, TrainConfig
from ipe_prediction.training.dataset import (
    SimulationTensors,
    ThermalDataset,
    excitation_mask,
    make_dataloader,
    prepare_data,
)
from ipe_prediction.training.engine import (
    LR_SCHEDULES,
    amp_dtype,
    evaluate,
    lr_scheduler,
    recalibrate_batchnorm,
    train_one_epoch,
)
from ipe_prediction.training.losses import LOSSES, n_network_outputs
from ipe_prediction.training.metrics import constraint_report, per_family_metrics
from ipe_prediction.training.models import MODELS, build_model, count_parameters
from ipe_prediction.training.preprocessing import Preprocessor, n_inputs

logger = logging.getLogger("ipe_prediction.train")

Data = tuple[SimulationTensors, dict[str, torch.Tensor]]

#: Schedules whose shape does not depend on ``epochs``: a finished run can be extended.
EXTENSIBLE_SCHEDULES = ("constant", "cosine_restarts", "step")


def subset(indices: torch.Tensor, n: int | None, seed: int) -> torch.Tensor:
    """Seeded random subset of ``n`` indices (all of them if ``n`` is None or 0)."""
    if not n or n >= len(indices):
        return indices
    perm = torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed))
    return indices[perm[:n]].sort().values


def _rank_and_world() -> tuple[int, int]:
    """Process rank and number of processes (0, 1 without DDP)."""
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


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
        resume: Continue from ``last.pt`` (same configuration, ``train.epochs`` may be raised).
        data: ``prepare_data(cfg.data)`` output, to avoid reloading it in a notebook.
    """
    run_dir = cfg.run_dir
    if (run_dir / "last.pt").exists() and not resume:
        raise FileExistsError(f"{run_dir} already holds a run: resume it or use another name")
    if _rank_and_world()[0] != 0:  # DDP: only process 0 logs and writes
        return _train(cfg, model, resume, data)
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
    rank, world = _rank_and_world()
    main_process = rank == 0
    if tc.batch_size % world:
        raise ValueError(f"batch_size {tc.batch_size} must be divisible by {world} GPUs")
    torch.manual_seed(tc.seed)
    device = torch.device(tc.device if torch.cuda.is_available() else "cpu")

    # --- data (seeded subsets: a resumed run sees the same simulations)
    tensors, split = data if data is not None else prepare_data(cfg.data)
    train_idx, val_idx = split["train"], split["val"]
    if cfg.data.min_duty_cycle > 0 or cfg.data.min_step_hours > 0:  # input-based filter
        keep = excitation_mask(tensors.heater_on, cfg.data.min_duty_cycle, cfg.data.min_step_hours)
        train_idx, val_idx = train_idx[keep[train_idx]], val_idx[keep[val_idx]]
        logger.info(
            "excitation filter (duty >= %g, mean step >= %g h): %d train, %d val simulations",
            cfg.data.min_duty_cycle, cfg.data.min_step_hours, len(train_idx), len(val_idx),
        )  # fmt: skip
    train_idx = subset(train_idx, cfg.data.max_train, tc.seed)
    val_idx = subset(val_idx, cfg.data.val_subset, tc.seed + 1)
    fit_idx = subset(train_idx, cfg.data.train_eval_subset, tc.seed + 2)
    prep = Preprocessor.fit(
        tensors.delta_t, tensors.targets, train_idx, cfg.data.extra_features, cfg.data.targets
    ).to(device)
    names = prep.target_names
    train_set = ThermalDataset(tensors, train_idx, cfg.data.temperature_noise_std)
    # each process reads its own shard; the shuffle depends on (seed, epoch) only
    sampler = DistributedSampler(train_set, num_replicas=world, rank=rank, seed=tc.seed)
    train_loader = make_dataloader(train_set, tc.batch_size // world, sampler=sampler)
    # same batches on every process, so that they all recompute the same BatchNorm statistics
    bn_loader = make_dataloader(train_set, tc.batch_size, shuffle=True, seed=tc.seed + 1)
    val_loader = make_dataloader(ThermalDataset(tensors, val_idx), tc.eval_batch_size)
    fit_loader = None
    if cfg.data.train_eval_subset:
        fit_loader = make_dataloader(ThermalDataset(tensors, fit_idx), tc.eval_batch_size)

    # --- model and optimisation
    if model is None:
        in_channels = n_inputs(cfg.data.extra_features)
        n_out = n_network_outputs(prep, tc.loss)
        model = build_model(cfg.model, cfg.model_config, in_channels, n_out)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    steps = len(train_loader)
    scheduler = lr_scheduler(
        optimizer,
        tc.lr_schedule,
        tc.epochs * steps,
        round(tc.warmup_epochs * steps),
        tc.min_lr / tc.lr,
        round(tc.restart_epochs * steps),
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
            "resumed after epoch %d (best %d, val %.4f), until epoch %d",
            ckpt["epoch"], best_epoch, best_val, tc.epochs,
        )  # fmt: skip
    if main_process:  # (re)written on resume: ``train.epochs`` may have been raised
        cfg.save(run_dir / "config.json")
        (run_dir / "model.txt").write_text(str(model))

    logger.info(
        "%s: %s parameters | train %d | val %d | train-eval %d | batch %d%s | %s | lr %s %.1e "
        "(min %.1e, warmup %.1f, cycle %.0f epochs) | weight decay %g | %d epochs",
        cfg.model,
        f"{count_parameters(model):,}",
        len(train_idx),
        len(val_idx),
        len(fit_idx) if fit_loader is not None else 0,
        tc.batch_size,
        f" ({world} GPUs x {tc.batch_size // world})" if world > 1 else "",
        str(dtype).replace("torch.", "autocast ") if dtype else "float32",
        tc.lr_schedule,
        tc.lr,
        tc.min_lr,
        tc.warmup_epochs,
        tc.restart_epochs,
        tc.weight_decay,
        tc.epochs,
    )
    # DDP synchronises the gradients; ``model`` stays the plain network (eval, checkpoints)
    trained = DistributedDataParallel(model) if world > 1 else model

    # --- epochs
    for epoch in range(start_epoch, tc.epochs + 1):
        t0 = time.time()
        sampler.set_epoch(epoch)  # same order when resuming
        train_loss = train_one_epoch(
            trained, prep, train_loader, optimizer, scheduler, grad_scaler, device, dtype,
            tc.grad_clip, tc.loss,
        )  # fmt: skip
        if world > 1:  # mean over the shards of all processes
            loss = torch.tensor(train_loss, device=device)
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            train_loss = loss.item()
        bn_loader.generator.manual_seed(tc.seed * 100_003 + epoch + 1)
        recalibrate_batchnorm(model, prep, bn_loader, device, dtype, tc.bn_recalibration_batches)
        val = evaluate(model, prep, val_loader, device, dtype, tc.loss)
        fit = evaluate(model, prep, fit_loader, device, dtype, tc.loss) if fit_loader else None
        if val["loss"] < best_val:
            best_val, best_epoch = val["loss"], epoch
            if main_process:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "preprocessor": prep.state_dict(),
                        "epoch": epoch,
                    },
                    run_dir / "best.pt",
                )
        if not main_process:
            continue

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "seconds": time.time() - t0,
            "lr": optimizer.param_groups[0]["lr"],
            **_metric_columns("val", val, names),
        }
        if fit is not None:
            row |= {"train_eval_loss": fit["loss"], **_metric_columns("train_eval", fit, names)}
        history.append(row)
        logger.info(
            "epoch %03d | train %.4f%s | val %.4f | val MAPE %% %s | val R2 %s | %.0fs | lr %.1e%s",
            epoch,
            train_loss,
            f" (eval {fit['loss']:.4f})" if fit is not None else "",
            val["loss"],
            " ".join(f"{n}={row[f'val_MAPE_{n}']:.2f}" for n in names),
            " ".join(f"{n}={row[f'val_R2_{n}']:.3f}" for n in names),
            row["seconds"],
            row["lr"],
            " *" if best_epoch == epoch else "",
        )
        if fit is not None:
            logger.info(
                "          train-eval R2 %s",
                " ".join(f"{n}={row[f'train_eval_R2_{n}']:.3f}" for n in names),
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
    if not main_process:
        return {}

    # --- final scores of the last epoch (overfit) and of the best one (kept in metrics_val.csv)
    summary: dict[str, Any] = {"best_epoch": best_epoch}
    for label, ckpt_name in (("last", "last.pt"), ("best", "best.pt")):
        model.load_state_dict(torch.load(run_dir / ckpt_name, weights_only=True)["model"])
        recalibrate_batchnorm(model, prep, bn_loader, device, dtype, tc.bn_recalibration_batches)
        val = evaluate(model, prep, val_loader, device, dtype, tc.loss)
        summary[f"{label}_val_loss"] = val["loss"]
        summary[f"{label}_val_R2"] = val["metrics"]["R2"].round(4).to_dict()
        if fit_loader is not None:
            fit = evaluate(model, prep, fit_loader, device, dtype, tc.loss)
            summary[f"{label}_train_eval_loss"] = fit["loss"]
            summary[f"{label}_train_eval_R2"] = fit["metrics"]["R2"].round(4).to_dict()
    # ``val`` now holds the best epoch
    val["metrics"].to_csv(run_dir / "metrics_val.csv")
    summary |= {
        "val_loss": val["loss"],
        "val_family": per_family_metrics(val["metrics"]).round(4).to_dict(orient="index"),
        "val_constraints": constraint_report(val["pred"], names),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        "validation metrics (best epoch %d):\n%s", best_epoch, val["metrics"].round(3).to_string()
    )
    return summary


def _metric_columns(prefix: str, res: dict[str, Any], names: Sequence[str]) -> dict[str, float]:
    """History columns of one evaluated split: loss, MAPE and R2 per target."""
    m, losses = res["metrics"], res["loss_per_target"]
    return {
        **{f"{prefix}_loss_{n}": float(v) for n, v in zip(names, losses, strict=True)},
        **{f"{prefix}_MAPE_{n}": float(m.loc[n, "MAPE_%"]) for n in names},
        **{f"{prefix}_R2_{n}": float(m.loc[n, "R2"]) for n in names},
    }


def _ddp_worker(rank: int, world: int, cfg: ExperimentConfig, resume: bool) -> None:
    """One DDP process per GPU (started by :func:`main` with ``--ddp``)."""
    backend = "nccl" if dist.is_nccl_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    if rank != 0:
        logging.getLogger("ipe_prediction").setLevel(logging.WARNING)
    try:
        run(cfg, resume=resume)
    finally:
        dist.destroy_process_group()


def _model_config(model: str, pairs: Sequence[str]) -> Any:
    """Default config of ``model`` with ``KEY=VALUE`` overrides (values read as Python literals)."""
    config = MODELS[model][0]()
    fields = {f.name for f in dataclasses.fields(config)}
    changes = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep or key not in fields:
            raise SystemExit(f"--model-set {pair!r}: expected KEY=VALUE, KEY in {sorted(fields)}")
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        changes[key] = tuple(value) if isinstance(value, list) else value
    return dataclasses.replace(config, **changes)


def parse_args(argv: Sequence[str] | None = None) -> tuple[ExperimentConfig, bool, bool]:
    """Configuration (or the saved one with ``--resume``), ``resume`` and ``ddp`` flags."""
    t, d = TrainConfig(), DataConfig()
    p = argparse.ArgumentParser(description="Train a model (train / validation splits only).")
    p.add_argument("--name", required=True, help="run name: outputs/runs/<name>")
    p.add_argument("--model", default="tcn_attn", choices=sorted(MODELS))
    p.add_argument(
        "--model-set",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="override the model config, e.g. --model-set dropout=0 channels=128",
    )
    p.add_argument("--resume", action="store_true", help="continue run <name> from last.pt")
    p.add_argument(
        "--epochs", type=int, help=f"default {t.epochs}; with --resume, extend the run to this"
    )
    p.add_argument("--batch-size", type=int, default=t.batch_size, help="global batch size")
    p.add_argument("--lr-schedule", choices=LR_SCHEDULES, default=t.lr_schedule)
    p.add_argument("--lr", type=float, default=t.lr, help="peak lr (after the warmup)")
    p.add_argument("--min-lr", type=float, default=t.min_lr, help="floor of the cosine schedules")
    p.add_argument("--warmup-epochs", type=float, default=t.warmup_epochs)
    p.add_argument(
        "--restart-epochs", type=float, default=t.restart_epochs, help="cosine_restarts cycle"
    )
    p.add_argument("--weight-decay", type=float, default=t.weight_decay)
    p.add_argument(
        "--loss", choices=LOSSES, default=t.loss, help="gaussian_nll: also predict a variance"
    )
    p.add_argument("--max-train", type=int, help="subsample the training split (quick runs)")
    p.add_argument(
        "--val-subset",
        type=int,
        default=d.val_subset,
        help="validation simulations scored each epoch (0 = the whole split)",
    )
    p.add_argument(
        "--train-eval-subset",
        type=int,
        default=d.train_eval_subset,
        help="training simulations also scored each epoch in eval mode (0 = off)",
    )
    p.add_argument("--min-duty-cycle", type=float, default=0.0, help="input filter (0 = off)")
    p.add_argument("--min-step-hours", type=float, default=0.0, help="input filter (0 = off)")
    p.add_argument(
        "--targets",
        nargs="+",
        choices=TARGET_NAMES,
        default=list(TARGET_NAMES),
        help="targets to learn (HLC = mit + ext only when all three are given)",
    )
    p.add_argument("--extra-features", action="store_true", help="5 input channels instead of 2")
    p.add_argument("--no-amp", dest="amp", action="store_false", help="train in float32")
    p.add_argument("--ddp", action="store_true", help="use all visible GPUs (Kaggle T4 x2)")
    a = p.parse_args(argv)

    if a.resume:
        cfg = ExperimentConfig.load(Path(ExperimentConfig.output_dir) / a.name)
        if a.epochs is not None and a.epochs != cfg.train.epochs:
            if cfg.train.lr_schedule not in EXTENSIBLE_SCHEDULES:
                raise SystemExit(
                    f"--epochs on resume needs a schedule in {EXTENSIBLE_SCHEDULES} "
                    f"(the {cfg.train.lr_schedule!r} schedule depends on the total)"
                )
            cfg.train.epochs = a.epochs
        return cfg, True, a.ddp
    cfg = ExperimentConfig(
        name=a.name,
        model=a.model,
        model_config=_model_config(a.model, a.model_set),
        data=DataConfig(
            max_train=a.max_train,
            val_subset=a.val_subset or None,
            train_eval_subset=a.train_eval_subset,
            extra_features=a.extra_features,
            targets=tuple(a.targets),
            min_duty_cycle=a.min_duty_cycle,
            min_step_hours=a.min_step_hours,
        ),
        train=TrainConfig(
            epochs=a.epochs if a.epochs is not None else t.epochs,
            batch_size=a.batch_size,
            lr_schedule=a.lr_schedule,
            lr=a.lr,
            min_lr=a.min_lr,
            warmup_epochs=a.warmup_epochs,
            restart_epochs=a.restart_epochs,
            weight_decay=a.weight_decay,
            loss=a.loss,
            amp=a.amp,
        ),
    )
    return cfg, False, a.ddp


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point (``uv run ipe-train``)."""
    cfg, resume, ddp = parse_args(argv)
    if not ddp:
        run(cfg, resume=resume)
        return
    world = torch.cuda.device_count()
    if world < 1:
        raise SystemExit("--ddp needs at least one GPU")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    torch.multiprocessing.spawn(_ddp_worker, args=(world, cfg, resume), nprocs=world)


if __name__ == "__main__":
    main()

"""Notebook helpers: define an architecture in a cell and train it with the standard pipeline.

The training loop, split, scaling, loss, checkpoints and metrics are exactly those of
``ipe-train`` (:func:`ipe_prediction.training.train.run`); only the network changes. A custom
network receives ``(B, F, L)`` scaled input channels and returns ``(B, 5)`` standardised outputs
(see :class:`ipe_prediction.training.models.CustomRegressor`)::

    from ipe_prediction.training import experiment as X

    class MyNet(nn.Module):
        def __init__(self, n_in=X.N_INPUTS): ...
        def forward(self, x):  # (B, 5, 480) -> (B, 5)
            ...

    res = X.train("my_net", MyNet, epochs=5, max_train=100_000, lr=3e-4)
    X.plot_histories(["my_net", "tcn"])

Any field of :class:`DataConfig`, :class:`ModelConfig` or :class:`TrainConfig` can be passed as a
keyword argument. The simulations are loaded once per session and reused by every run.
"""

from __future__ import annotations

import inspect
import json
import logging
import shutil
from collections.abc import Callable, Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn

from ipe_prediction.training.config import (
    CUSTOM_ARCH,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
)
from ipe_prediction.training.dataset import SimulationTensors, prepare_data
from ipe_prediction.training.models import (
    CustomRegressor,
    ThermalRegressor,
    build_model,
    n_input_channels,
)
from ipe_prediction.training.preprocessing import TargetScaler
from ipe_prediction.training.train import BEST, LAST, run

logger = logging.getLogger(__name__)

#: Input channels a custom network receives with the default ``tcn_extra_features=True``:
#: ``[delta_t / std, heater_on, t / L, cumsum(heater_on) / L, cumsum(delta_t / std) / L]``.
N_INPUTS: int = n_input_channels(False)
#: Outputs a network must return: standardised ``[HLC_mit, HLC_ext, log tau_b, log tau_n,
#: log tau_inf]``.
N_OUTPUTS: int = 5

_DATA: dict[tuple[Any, ...], tuple[SimulationTensors, dict[str, torch.Tensor]]] = {}


def load_data(cfg: DataConfig | None = None) -> tuple[SimulationTensors, dict[str, torch.Tensor]]:
    """All simulations and the frozen split, loaded once per session (about 2.4 GB of RAM)."""
    cfg = cfg or DataConfig()
    key = (cfg.folder, cfg.quantize, str(cfg.split_path), cfg.split_fractions)
    if key not in _DATA:
        _DATA.clear()  # keep a single copy in memory
        _DATA[key] = prepare_data(cfg)
    return _DATA[key]


def make_config(name: str, arch: str = "tcn_attn", **overrides: Any) -> ExperimentConfig:
    """Experiment configuration with keyword overrides routed to the matching sub-config."""
    groups: dict[str, dict[str, Any]] = {"data": {}, "model": {"arch": arch}, "train": {}}
    top: dict[str, Any] = {}
    owners = {
        **{f.name: "data" for f in fields(DataConfig)},
        **{f.name: "model" for f in fields(ModelConfig)},
        **{f.name: "train" for f in fields(TrainConfig)},
    }
    for key, value in overrides.items():
        if key == "output_dir":
            top[key] = value
        elif key in owners:
            groups[owners[key]][key] = value
        else:
            raise TypeError(f"unknown setting {key!r}; valid: {sorted(owners)}")
    return ExperimentConfig(
        name=name,
        data=DataConfig(**groups["data"]),
        model=ModelConfig(**groups["model"]),
        train=TrainConfig(**groups["train"]),
        **top,
    )


def _network_source(net: Callable[[], nn.Module]) -> str:
    try:
        return inspect.getsource(net)
    except (OSError, TypeError):
        return repr(net)


def train(
    name: str,
    net: Callable[[], nn.Module] | None = None,
    *,
    arch: str = "tcn_attn",
    epochs: int = 10,
    resume: bool = False,
    overwrite: bool = False,
    **overrides: Any,
) -> dict[str, Any]:
    """Train one experiment and return its summary (plus ``"model"`` and ``"history"``).

    Args:
        name: Run name; artefacts go to ``outputs/runs/<name>/``.
        net: Factory of a custom network (a class or a ``lambda``) mapping ``(B, F, L)`` to
            ``(B, 5)``; ``F`` is :data:`N_INPUTS` (2 with ``tcn_extra_features=False``).
            ``None`` trains the built-in ``arch`` (``"tcn_attn"`` or ``"cnn_gru"``).
        arch: Built-in architecture when ``net`` is None.
        epochs: Total number of epochs (with ``resume``, the new total).
        resume: Continue the existing run ``name`` (same settings, pass the same ``net``).
        overwrite: Delete an existing run ``name`` and start again.
        **overrides: Any ``DataConfig`` / ``ModelConfig`` / ``TrainConfig`` field, e.g.
            ``max_train=100_000, lr=3e-4, batch_size=256, amp=True, tcn_channels=96``, or
            ``output_dir``.
    """
    cfg = make_config(name, CUSTOM_ARCH if net is not None else arch, epochs=epochs, **overrides)
    run_dir = cfg.run_dir
    if resume:
        if not (run_dir / LAST).exists():
            raise FileNotFoundError(f"nothing to resume: no {LAST} in {run_dir}")
        cfg = ExperimentConfig.from_dict(json.loads((run_dir / "config.json").read_text()))
        cfg.train.epochs = epochs
    elif (run_dir / LAST).exists():
        if not overwrite:
            raise FileExistsError(f"{run_dir} already holds a run: resume=True or overwrite=True")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if net is not None:
        (run_dir / "network.py.txt").write_text(_network_source(net))

    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root.setLevel(logging.INFO)
    handler = logging.FileHandler(run_dir / "train.log", mode="a" if resume else "w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root.addHandler(handler)
    try:
        summary = run(cfg, resume=resume, net=net, data=load_data(cfg.data))
    finally:
        root.removeHandler(handler)
        handler.close()
    return summary | {"history": pd.read_csv(run_dir / "history.csv")}


def load_model(
    name: str,
    net: Callable[[], nn.Module] | None = None,
    output_dir: str = ExperimentConfig.output_dir,
    device: str = "cpu",
) -> ThermalRegressor:
    """Best checkpoint of run ``name``; custom runs need the same ``net`` factory again."""
    ckpt = torch.load(Path(output_dir) / name / BEST, map_location=device, weights_only=False)
    cfg = ExperimentConfig.from_dict(ckpt["config"])
    if cfg.model.arch == CUSTOM_ARCH:
        if net is None:
            raise ValueError(f"run {name!r} uses a custom network: pass its factory as net")
        model: ThermalRegressor = CustomRegressor(
            net(), 1.0, TargetScaler.empty(), cfg.model.tcn_extra_features
        )
    else:
        model = build_model(cfg.model, 1.0, TargetScaler.empty())
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


def histories(names: Sequence[str], output_dir: str = ExperimentConfig.output_dir) -> pd.DataFrame:
    """Concatenated ``history.csv`` of several runs (column ``run``)."""
    frames = [
        pd.read_csv(Path(output_dir) / n / "history.csv").assign(run=n)
        for n in names
        if (Path(output_dir) / n / "history.csv").exists()
    ]
    return pd.concat(frames, ignore_index=True)


def compare(names: Sequence[str], output_dir: str = ExperimentConfig.output_dir) -> pd.DataFrame:
    """One row per run: epochs, best validation loss and the validation MAPE at that epoch."""
    hist = histories(names, output_dir)
    best = hist.loc[hist.groupby("run")["val_loss"].idxmin()].set_index("run")
    cols = ["epoch", "val_loss", "train_loss", "seconds"]
    cols += [c for c in hist.columns if c.startswith("val_MAPE_")]
    table = best[cols].rename(columns={"epoch": "best_epoch", "seconds": "s_per_epoch"})
    table.insert(0, "epochs", hist.groupby("run")["epoch"].max())
    return table.sort_values("val_loss")


def plot_histories(
    names: Sequence[str],
    metrics: Sequence[str] = ("train_loss", "val_loss"),
    output_dir: str = ExperimentConfig.output_dir,
) -> plt.Figure:
    """Learning curves of several runs, one panel per metric."""
    hist = histories(names, output_dir)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 3.5), squeeze=False)
    for ax, metric in zip(axes[0], metrics, strict=True):
        for run_name, h in hist.groupby("run", sort=False):
            ax.plot(h["epoch"], h[metric], marker=".", label=run_name)
        ax.set(xlabel="epoch", title=metric)
        ax.grid(alpha=0.3)
    axes[0][0].legend()
    fig.tight_layout()
    return fig

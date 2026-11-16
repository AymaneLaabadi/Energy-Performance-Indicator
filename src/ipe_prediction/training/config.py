"""Experiment configuration: data, training and the model (whose config lives in its own file)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ipe_prediction.training.models import MODELS


@dataclass
class DataConfig:
    """Data location, frozen split and input options."""

    folder: str = "quentin_house"
    split_dir: str = "outputs/splits"
    split_seed: int = 42
    split_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)
    #: Apply the assessment's rounding (0.01 degC, 0.1 W).
    quantize: bool = True
    #: Add the time / cumulative-heating / integrated-rise input channels (see preprocessing).
    extra_features: bool = False
    #: Std of Gaussian noise added to the training temperatures (K); 0 disables it.
    temperature_noise_std: float = 0.0
    #: Random subset of the training split (quick runs); None = all of it.
    max_train: int | None = None
    #: Random subset of the validation split scored each epoch; None = all of it.
    val_subset: int | None = 10_000

    @property
    def split_path(self) -> Path:
        return Path(self.split_dir) / f"split_seed{self.split_seed}.npz"


@dataclass
class TrainConfig:
    """AdamW with linear warmup then cosine decay (per batch) down to ``min_lr``."""

    epochs: int = 60
    batch_size: int = 512
    eval_batch_size: int = 1024
    lr: float = 1e-3
    min_lr: float = 1e-6
    warmup_epochs: float = 1.0
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    #: Mixed precision on CUDA: bf16 on Ampere+ GPUs, fp16 with gradient scaling otherwise.
    amp: bool = True
    seed: int = 0
    device: str = "cuda"


@dataclass
class ExperimentConfig:
    """A run: its name, the model (registered name + its config), data and training settings.

    ``model_config`` defaults to the registered model's config class. For a model defined outside
    the registry (e.g. in a notebook) ``model`` is just a label and ``model_config`` stays None.
    """

    name: str
    model: str = "tcn_attn"
    model_config: Any = None
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dir: str = "outputs/runs"

    def __post_init__(self) -> None:
        if self.model_config is None and self.model in MODELS:
            self.model_config = MODELS[self.model][0]()

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # recurses into model_config

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentConfig:
        def tuples(sub: dict[str, Any]) -> dict[str, Any]:
            return {k: tuple(v) if isinstance(v, list) else v for k, v in sub.items()}

        model_config = None
        if d["model"] in MODELS and d.get("model_config") is not None:
            model_config = MODELS[d["model"]][0](**tuples(d["model_config"]))
        return cls(
            name=d["name"],
            model=d["model"],
            model_config=model_config,
            data=DataConfig(**tuples(d["data"])),
            train=TrainConfig(**tuples(d["train"])),
            output_dir=d["output_dir"],
        )

    @classmethod
    def load(cls, run_dir: Path | str) -> ExperimentConfig:
        """Configuration saved in ``<run_dir>/config.json``."""
        return cls.from_dict(json.loads((Path(run_dir) / "config.json").read_text()))

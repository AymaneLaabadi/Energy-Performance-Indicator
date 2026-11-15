"""Experiment configuration as typed, serialisable dataclasses."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    """Data location and the frozen, stratified 80/10/10 split (see :mod:`ipe_prediction.training.splits`)."""

    folder: str = "quentin_house"
    split_seed: int = 42
    split_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)
    split_dir: str = "outputs/splits"
    quantize: bool = True
    #: Std of Gaussian noise added to the temperature during training (K). 0 disables it.
    temperature_noise_std: float = 0.0
    #: Random subset of the training split (quick runs); ``None`` uses all of it.
    max_train: int | None = None
    #: Random subset of the validation split scored every epoch; ``None`` uses all of it.
    #: ``ipe-evaluate`` always scores the full split.
    val_subset: int | None = 10000

    @property
    def split_path(self) -> Path:
        """File holding the frozen split indices."""
        return Path(self.split_dir) / f"split_seed{self.split_seed}.npz"


#: Architectures known to :func:`ipe_prediction.training.models.build_model`.
ARCHITECTURES: tuple[str, ...] = ("tcn_attn", "cnn_gru")
#: ``arch`` of a network defined outside the package (e.g. in a notebook), see
#: :func:`ipe_prediction.training.experiment.train`.
CUSTOM_ARCH: str = "custom"


@dataclass
class ModelConfig:
    """Model choice (``arch``) and the hyper-parameters of each architecture.

    * ``cnn_gru``: :class:`ipe_prediction.training.models.BaselineCNNGRU` (``conv_*``, ``gru_*``);
    * ``tcn_attn``: :class:`ipe_prediction.training.models.TCNAttention` (``tcn_*``, ``attn_*``);
    * ``custom``: a network passed to :func:`ipe_prediction.training.experiment.train` (only
      ``tcn_extra_features`` applies: it sets the input channels it receives).

    ``mlp_hidden`` is the hidden width of the regression head of both.
    """

    arch: str = "tcn_attn"
    # --- cnn_gru
    conv_channels: tuple[int, ...] = (32, 64, 64)
    conv_kernels: tuple[int, ...] = (5, 5, 5)
    conv_strides: tuple[int, ...] = (2, 2, 2)
    gru_hidden: int = 32
    gru_layers: int = 2
    
    # --- tcn_attn
    #: Width of the residual TCN blocks.
    tcn_channels: int = 64
    tcn_kernel: int = 3
    #: One residual block (two dilated convs) per dilation. With kernel 3 and stem stride 2 the
    #: receptive field is ``1 + 4 * sum(dilations)`` = 509 steps of 2 min, i.e. the whole 8 h.
    tcn_dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    #: Stride of the stem conv (2 -> 240 steps of 2 min; halves time and memory).
    tcn_stem_stride: int = 2
    tcn_dropout: float = 0.1
    #: Add physics-motivated input channels (elapsed time, cumulative heating, integral of the
    #: rise) to the two raw series.
    tcn_extra_features: bool = False
    #: Number of attention-pooling heads (must divide ``tcn_channels``).
    attn_heads: int = 4
    # --- shared
    mlp_hidden: int = 64

    def __post_init__(self) -> None:
        if self.arch not in (*ARCHITECTURES, CUSTOM_ARCH):
            raise ValueError(
                f"unknown arch {self.arch!r}; expected one of {(*ARCHITECTURES, CUSTOM_ARCH)}"
            )


@dataclass
class TrainConfig:
    """Optimisation settings: AdamW with a fixed learning rate, a fixed number of epochs.

    The best validation epoch is kept as ``best.pt``; ``last.pt`` holds the full training state so
    a stopped run can be resumed (``ipe-train --name <run> --resume``).
    """

    epochs: int = 500
    batch_size: int = 512
    eval_batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    #: bfloat16 autocast on CUDA (about 2x faster for the GRU; no loss scaling needed).
    amp: bool = True
    seed: int = 0
    device: str = "cuda"


@dataclass
class ExperimentConfig:
    """Full experiment: data, model, training and output location."""

    name: str = "tcn_attn"
    output_dir: str = "outputs/runs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def run_dir(self) -> Path:
        """Directory where all artefacts of this run are written."""
        return Path(self.output_dir) / self.name

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view (JSON serialisable)."""
        return asdict(self)

    def save(self, path: Path) -> None:
        """Write the configuration as JSON."""
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentConfig:
        """Rebuild a configuration from :meth:`to_dict` output."""

        def _tuples(sub: dict[str, Any]) -> dict[str, Any]:
            return {k: tuple(v) if isinstance(v, list) else v for k, v in sub.items()}

        return cls(
            name=d["name"],
            output_dir=d["output_dir"],
            data=DataConfig(**_tuples(d["data"])),
            # runs saved before ``arch`` existed are CNN-GRU runs
            model=ModelConfig(**({"arch": "cnn_gru"} | _tuples(d["model"]))),
            train=TrainConfig(**_tuples(d["train"])),
        )

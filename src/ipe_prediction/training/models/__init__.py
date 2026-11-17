"""Model registry: one file per architecture, each with its own config dataclass.

Every model maps ``(B, C_in, L)`` inputs (see :mod:`ipe_prediction.training.preprocessing`) to
``(B, n_outputs)`` standardised outputs (5 with all targets), and is built as
``Model(config, in_channels, n_outputs)``.

To add a model: write ``models/<name>.py`` with a config dataclass and an ``nn.Module``, then
register both in :data:`MODELS`.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from ipe_prediction.training.models.cnn_gru import CNNGRU, CNNGRUConfig
from ipe_prediction.training.models.model_a import ModelA, ModelAConfig
from ipe_prediction.training.models.tcn_attn import TCNAttention, TCNAttentionConfig
from ipe_prediction.training.preprocessing import N_OUTPUTS

#: name -> (config class, model class)
MODELS: dict[str, tuple[type, type[nn.Module]]] = {
    "cnn_gru": (CNNGRUConfig, CNNGRU),
    "tcn_attn": (TCNAttentionConfig, TCNAttention),
    "model_a": (ModelAConfig, ModelA),
}


def build_model(name: str, config: Any, in_channels: int, n_outputs: int = N_OUTPUTS) -> nn.Module:
    """Instantiate the registered model ``name``."""
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; registered: {sorted(MODELS)}")
    return MODELS[name][1](config, in_channels, n_outputs)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

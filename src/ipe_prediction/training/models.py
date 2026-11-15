"""Neural network models.

Every model maps ``delta_t`` ``(B, L)`` (indoor temperature rise, K) and ``heater_on`` ``(B, L)``
(0/1) to ``(B, 5)`` standardised outputs ``[HLC_mit, HLC_ext, log tau_b, log tau_n, log tau_inf]``
(see :class:`~ipe_prediction.training.preprocessing.TargetScaler`; ``HLC`` is their mit + ext sum).

* :class:`BaselineCNNGRU`: conv front end + bidirectional GRU (the baseline);
* :class:`TCNAttention`: dilated residual TCN + multi-head attention pooling;
* :class:`CustomRegressor`: wraps any network defined elsewhere (e.g. in a notebook).
"""

from __future__ import annotations

import torch
from torch import nn

from ipe_prediction.training.config import ModelConfig
from ipe_prediction.training.preprocessing import TargetScaler

#: Network outputs: ``[HLC_mit, HLC_ext, tau_b, tau_n, tau_inf]`` (HLC is their mit + ext sum).
N_OUTPUTS: int = 5


class ThermalRegressor(nn.Module):
    """Common part of all models: input scale and target scaler stored as buffers.

    Storing them in the module makes a checkpoint self-contained.
    """

    def __init__(self, delta_t_std: float, target_scaler: TargetScaler) -> None:
        super().__init__()
        self.register_buffer("delta_t_std", torch.tensor(float(delta_t_std)))
        self.target_scaler = target_scaler

    def forward(self, delta_t: torch.Tensor, heater_on: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, delta_t: torch.Tensor, heater_on: torch.Tensor) -> torch.Tensor:
        """Physical predictions ``(B, 6)`` in the canonical target order."""
        return self.target_scaler.decode(self(delta_t, heater_on))


# ---------------------------------------------------------------------------------------------
# Baseline CNN-GRU
# ---------------------------------------------------------------------------------------------


class ConvBlock(nn.Sequential):
    """``Conv1d -> GELU`` with 'same'-style padding (no normalisation)."""

    def __init__(self, c_in: int, c_out: int, kernel: int, stride: int) -> None:
        super().__init__(
            nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=kernel // 2, bias=False),
            nn.GELU(),
        )


class BaselineCNNGRU(ThermalRegressor):
    """Baseline CNN-GRU regressor.

    Architecture (sizes from ``ModelConfig.conv_*`` / ``gru_*``)::

        [delta_t / std, heater_on]                       (B, 2, L)
        Conv1d + GELU, one per conv_channels entry       (B, C, L')
        bidirectional GRU                                final states (B, 2 x gru_hidden)
        Linear(-> mlp_hidden) + GELU + Linear(-> 5)      (B, 5)
    """

    def __init__(self, cfg: ModelConfig, delta_t_std: float, target_scaler: TargetScaler) -> None:
        super().__init__(delta_t_std, target_scaler)
        blocks, c_in = [], 2
        for c_out, k, s in zip(cfg.conv_channels, cfg.conv_kernels, cfg.conv_strides, strict=True):
            blocks.append(ConvBlock(c_in, c_out, k, s))
            c_in = c_out
        self.conv = nn.Sequential(*blocks)
        self.gru = nn.GRU(
            input_size=c_in,
            hidden_size=cfg.gru_hidden,
            num_layers=cfg.gru_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * cfg.gru_hidden, cfg.mlp_hidden),
            nn.GELU(),
            nn.Linear(cfg.mlp_hidden, N_OUTPUTS),
        )

    def forward(self, delta_t: torch.Tensor, heater_on: torch.Tensor) -> torch.Tensor:
        """``(B, L)`` rise in K and ``(B, L)`` heater state in {0, 1} -> ``(B, 5)`` outputs."""
        x = torch.stack([delta_t / self.delta_t_std, heater_on], dim=1)  # (B, 2, L)
        h = self.conv(x).transpose(1, 2)  # (B, L', C)
        _, h_n = self.gru(h)  # (layers * 2, B, H)
        last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # final forward + backward states, top layer
        return self.mlp(last)


# ---------------------------------------------------------------------------------------------
# TCN + attention pooling
# ---------------------------------------------------------------------------------------------


def physical_features(
    delta_t: torch.Tensor, heater_on: torch.Tensor, delta_t_std: torch.Tensor, extra: bool
) -> torch.Tensor:
    """Input channels ``(B, F, L)`` built from the two raw series.

    Always: ``delta_t / std`` and ``heater_on``. With ``extra`` (all of order 1):

    * elapsed time ``t / L`` (attention pooling is order-free; this tells it *when*);
    * cumulative heating ``cumsum(heater_on) / L`` (injected energy, in units of ``P_MAX x 8 h``);
    * integral of the rise ``cumsum(delta_t / std) / L``.

    The last two make the lumped energy balance ``C dT = U dt - H T dt`` linear in the inputs:
    integrated over ``[0, t]`` it reads ``C T(t) = int U - H int T``, which relates the three
    channels through the very quantities to predict (``H``, ``C`` and thus ``tau = C / H``).
    """
    x = delta_t / delta_t_std
    channels = [x, heater_on]
    if extra:
        length = delta_t.shape[1]
        t = torch.linspace(0.0, 1.0, length, device=delta_t.device, dtype=delta_t.dtype)
        channels += [
            t.expand_as(delta_t),
            heater_on.cumsum(1) / length,
            x.cumsum(1) / length,
        ]
    return torch.stack(channels, dim=1)


class ResidualTCNBlock(nn.Module):
    """Pre-activation residual block: ``x + Drop(Conv_d(GELU(BN(Conv_d(GELU(BN(x)))))))``.

    Non-causal ('same') dilated convolutions: the whole 8 h window is available at inference, so
    each step may look both ways. BatchNorm (not Group/LayerNorm) on purpose: per-sample
    normalisation would erase the absolute amplitude of the rise, which carries ``HLC``.
    """

    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.body = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class AttentionPooling(nn.Module):
    """Multi-head additive attention pooling over time: ``(B, C, T) -> (B, C)``.

    Each head ``h`` scores every step, ``a_h(t) = softmax_t(w_h . tanh(W x_t))``, and returns the
    weighted sum of its slice of a value projection; the heads are concatenated. Different heads
    can focus on different parts of the transient (heating steps, free decay, the end of the
    window). The softmax runs in float32 for stability under bf16 autocast.
    """

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError(f"attn_heads={heads} must divide tcn_channels={channels}")
        self.heads = heads
        self.score = nn.Sequential(
            nn.Linear(channels, channels // 2), nn.Tanh(), nn.Linear(channels // 2, heads)
        )
        self.value = nn.Linear(channels, channels)

    def weights(self, x: torch.Tensor) -> torch.Tensor:
        """Attention weights ``(B, T, heads)`` (sum to 1 over ``T``)."""
        return self.score(x.transpose(1, 2)).float().softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)  # (B, T, C)
        a = self.weights(x)  # (B, T, heads)
        v = self.value(h).float().unflatten(-1, (self.heads, -1))  # (B, T, heads, C / heads)
        return torch.einsum("bth,bthd->bhd", a, v).flatten(1).to(x.dtype)  # (B, C)


class TCNAttention(ThermalRegressor):
    """Dilated residual TCN with multi-head attention pooling.

    Architecture (defaults of :class:`ModelConfig`)::

        physical_features                               (B, 5, 480)
        stem Conv1d(k=5, stride 2)                      (B, 64, 240)   1 step = 2 min
        7 residual blocks, dilations 1 ... 64, k=3      (B, 64, 240)   receptive field 509 > 240
        BN + GELU, attention pooling (4 heads)          (B, 64)
        Linear(-> mlp_hidden) + GELU + Dropout + Linear (B, 5)

    Unlike the GRU, all time steps are processed in parallel (fast on GPU, memory linear in
    ``L``), and the receptive field covering the whole window lets each step's features describe
    the full response; the pooling then decides which steps matter for each target.
    """

    def __init__(self, cfg: ModelConfig, delta_t_std: float, target_scaler: TargetScaler) -> None:
        super().__init__(delta_t_std, target_scaler)
        self.extra_features = cfg.tcn_extra_features
        c, s = cfg.tcn_channels, cfg.tcn_stem_stride
        n_in = 5 if cfg.tcn_extra_features else 2
        self.stem = nn.Conv1d(n_in, c, kernel_size=2 * s + 1, stride=s, padding=s)
        self.blocks = nn.Sequential(
            *(ResidualTCNBlock(c, cfg.tcn_kernel, d, cfg.tcn_dropout) for d in cfg.tcn_dilations)
        )
        self.out_norm = nn.Sequential(nn.BatchNorm1d(c), nn.GELU())
        self.pool = AttentionPooling(c, cfg.attn_heads)
        self.head = nn.Sequential(
            nn.Linear(c, cfg.mlp_hidden),
            nn.GELU(),
            nn.Dropout(cfg.tcn_dropout),
            nn.Linear(cfg.mlp_hidden, N_OUTPUTS),
        )

    def features(self, delta_t: torch.Tensor, heater_on: torch.Tensor) -> torch.Tensor:
        """Per-step features ``(B, C, T)`` fed to the attention pooling."""
        x = physical_features(delta_t, heater_on, self.delta_t_std, self.extra_features)
        return self.out_norm(self.blocks(self.stem(x)))

    def forward(self, delta_t: torch.Tensor, heater_on: torch.Tensor) -> torch.Tensor:
        """``(B, L)`` rise in K and ``(B, L)`` heater state in {0, 1} -> ``(B, 5)`` outputs."""
        return self.head(self.pool(self.features(delta_t, heater_on)))

    @torch.no_grad()
    def attention_weights(self, delta_t: torch.Tensor, heater_on: torch.Tensor) -> torch.Tensor:
        """Attention weights ``(B, T, heads)`` on the stem's time grid (for interpretation)."""
        return self.pool.weights(self.features(delta_t, heater_on))


# ---------------------------------------------------------------------------------------------
# Networks defined outside the package
# ---------------------------------------------------------------------------------------------


def n_input_channels(extra_features: bool) -> int:
    """Number of channels produced by :func:`physical_features`."""
    return 5 if extra_features else 2


class CustomRegressor(ThermalRegressor):
    """Wrap a user network ``net: (B, F, L) -> (B, 5)`` into a :class:`ThermalRegressor`.

    The wrapper builds the scaled input channels (:func:`physical_features`, ``F`` =
    :func:`n_input_channels`) and holds the scaling statistics, so ``net`` only has to map
    ``(B, F, 480)`` to the 5 standardised outputs.
    """

    def __init__(
        self,
        net: nn.Module,
        delta_t_std: float,
        target_scaler: TargetScaler,
        extra_features: bool = True,
    ) -> None:
        super().__init__(delta_t_std, target_scaler)
        self.net = net
        self.extra_features = extra_features

    def forward(self, delta_t: torch.Tensor, heater_on: torch.Tensor) -> torch.Tensor:
        return self.net(
            physical_features(delta_t, heater_on, self.delta_t_std, self.extra_features)
        )


MODELS: dict[str, type[ThermalRegressor]] = {"cnn_gru": BaselineCNNGRU, "tcn_attn": TCNAttention}


def build_model(cfg: ModelConfig, delta_t_std: float, scaler: TargetScaler) -> ThermalRegressor:
    """Instantiate ``cfg.arch`` from its configuration and training statistics."""
    if cfg.arch not in MODELS:
        raise ValueError(
            f"arch {cfg.arch!r} is not built from the config: rebuild the network and use "
            "ipe_prediction.training.experiment.load_model(name, net)"
        )
    return MODELS[cfg.arch](cfg, delta_t_std, scaler)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

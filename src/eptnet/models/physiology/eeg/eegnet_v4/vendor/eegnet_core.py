"""Dependency-light EEGNet-v4 core.

Adapted from ``braindecode.models.EEGNet`` at commit
15561d191db75c5e1360cb8573c18eacdef95b54 under BSD-3-Clause.  The model
topology and defaults follow Lawhern et al. (2018); package-specific mixins,
Hub integration and deprecated argument adapters were intentionally removed.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.parametrize import register_parametrization


class _MaxNorm(nn.Module):
    def __init__(self, maximum: float) -> None:
        super().__init__()
        self.maximum = float(maximum)

    def forward(self, weights: Tensor) -> Tensor:
        return weights.renorm(p=2, dim=0, maxnorm=self.maximum)


class _ConstrainedConv2d(nn.Conv2d):
    def __init__(self, *args, max_norm: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        nn.init.xavier_uniform_(self.weight, gain=1.0)
        register_parametrization(self, "weight", _MaxNorm(max_norm))


class EEGNetV4(nn.Module):
    """Compact EEGNet-v4 classifier with an explicit feature interface.

    Inputs are ``[batch, channels, samples]``. ``forward_features`` returns the
    flattened pre-classifier representation and ``forward`` returns logits.
    The canonical EPT baseline uses 8 channels, 200 samples and 2 classes,
    resulting in 96 features and 1,426 total parameters.
    """

    def __init__(
        self,
        n_chans: int = 8,
        n_times: int = 200,
        n_outputs: int = 2,
        *,
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int | None = None,
        kernel_length: int = 64,
        depthwise_kernel_length: int = 16,
        pool1_kernel_size: int = 4,
        pool2_kernel_size: int = 8,
        drop_prob: float = 0.25,
        spatial_max_norm: float = 1.0,
    ) -> None:
        super().__init__()
        if min(n_chans, n_times, n_outputs, f1, depth_multiplier) <= 0:
            raise ValueError("EEGNet dimensions must be positive")
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError("drop_prob must lie in [0, 1)")
        f2 = f1 * depth_multiplier if f2 is None else int(f2)
        self.n_chans = int(n_chans)
        self.n_times = int(n_times)
        self.n_outputs = int(n_outputs)
        self.f2 = f2

        self.feature_layers = nn.Sequential(
            nn.Conv2d(
                1,
                f1,
                kernel_size=(1, kernel_length),
                bias=False,
                padding=(0, kernel_length // 2),
            ),
            nn.BatchNorm2d(f1, momentum=0.01, affine=True, eps=1e-3),
            _ConstrainedConv2d(
                f1,
                f1 * depth_multiplier,
                kernel_size=(self.n_chans, 1),
                max_norm=spatial_max_norm,
                bias=False,
                groups=f1,
            ),
            nn.BatchNorm2d(f1 * depth_multiplier, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool1_kernel_size)),
            nn.Dropout(p=drop_prob),
            nn.Conv2d(
                f1 * depth_multiplier,
                f1 * depth_multiplier,
                kernel_size=(1, depthwise_kernel_length),
                bias=False,
                groups=f1 * depth_multiplier,
                padding=(0, depthwise_kernel_length // 2),
            ),
            nn.Conv2d(
                f1 * depth_multiplier,
                f2,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(f2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool2_kernel_size)),
            nn.Dropout(p=drop_prob),
        )

        was_training = self.feature_layers.training
        self.feature_layers.eval()
        with torch.no_grad():
            feature_map = self.feature_layers(torch.zeros(1, 1, self.n_chans, self.n_times))
        self.feature_layers.train(was_training)
        self.feature_map_shape = tuple(int(value) for value in feature_map.shape[1:])
        self.feature_dim = int(feature_map[0].numel())
        self.classifier = nn.Conv2d(
            f2,
            self.n_outputs,
            kernel_size=(self.feature_map_shape[1], self.feature_map_shape[2]),
            bias=True,
        )
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                weight = getattr(module, "weight", None)
                if weight is not None:
                    nn.init.xavier_uniform_(weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                module.reset_running_stats()

    def _validate_input(self, eeg: Tensor) -> None:
        if eeg.ndim != 3:
            raise ValueError("EEGNet input must have shape [batch, channels, samples]")
        if eeg.shape[1:] != (self.n_chans, self.n_times):
            raise ValueError(
                f"Expected EEG shape [B, {self.n_chans}, {self.n_times}], got {tuple(eeg.shape)}"
            )
        if not eeg.is_floating_point():
            raise TypeError("EEGNet input must be floating point")

    def forward_feature_map(self, eeg: Tensor) -> Tensor:
        self._validate_input(eeg)
        return self.feature_layers(eeg.unsqueeze(1))

    def forward_features(self, eeg: Tensor) -> Tensor:
        return self.forward_feature_map(eeg).flatten(start_dim=1)

    def forward(self, eeg: Tensor) -> Tensor:
        logits = self.classifier(self.forward_feature_map(eeg))
        return logits.flatten(start_dim=1)

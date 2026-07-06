"""2-D CNN over (time x subcarrier) with receivers as input channels.

The (B, n_rx, T, S) window tensor is treated as an image whose channels are
the receivers, so the receiver-count ablation (1 vs 2 vs 3 boards) is just a
different ``n_rx``. Every pooling stage uses ``ceil_mode`` and the trunk ends
in adaptive average pooling, so small windows (T >= 64, S >= 32 — the 1 s /
25 Hz ablation corner) never collapse to a zero-sized feature map.
"""

from __future__ import annotations

import torch
from torch import nn

# Channel widths of the four conv blocks; ~240k parameters total.
_CHANNELS: tuple[int, ...] = (32, 64, 128, 128)


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, ceil_mode=True),
    )


class CSICnn(nn.Module):
    """Conv blocks -> global average pool -> dropout -> linear logits."""

    def __init__(
        self,
        n_rx: int,
        n_time: int,
        n_subcarriers: int,
        n_classes: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if n_rx < 1:
            raise ValueError(f"n_rx must be >= 1, got {n_rx}")
        if n_time < 1 or n_subcarriers < 1:
            raise ValueError(
                f"n_time and n_subcarriers must be >= 1, "
                f"got {n_time} and {n_subcarriers}"
            )
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")
        self.n_rx = n_rx
        self.n_time = n_time
        self.n_subcarriers = n_subcarriers
        self.n_classes = n_classes

        widths = (n_rx, *_CHANNELS)
        self.features = nn.Sequential(
            *[_conv_block(widths[i], widths[i + 1]) for i in range(len(_CHANNELS))]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(_CHANNELS[-1], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_rx, T, S) float -> (B, n_classes) logits."""
        if x.ndim != 4 or x.shape[1] != self.n_rx:
            raise ValueError(
                f"expected (B, {self.n_rx}, T, S), got shape {tuple(x.shape)}"
            )
        z = self.features(x)
        z = self.pool(z).flatten(1)
        return self.head(self.dropout(z))

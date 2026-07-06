"""CNN encoder per time chunk -> bidirectional LSTM -> classification head.

The window's time axis is split into fixed-length chunks (default 25 samples
= 0.25 s at 100 Hz, so a 3 s window becomes 12 steps). A shared-weight CNN
encodes each (n_rx, chunk_len, S) chunk to a feature vector; a BiLSTM models
the chunk sequence; mean pooling over time feeds the classifier. A trailing
remainder (T not divisible by chunk_len) is trimmed.
"""

from __future__ import annotations

import torch
from torch import nn

# Chunk-encoder conv widths; the last entry is the LSTM input size.
_ENCODER_CHANNELS: tuple[int, ...] = (32, 64)


class _ChunkEncoder(nn.Module):
    """Two conv blocks + global average pool: (B, n_rx, chunk_len, S) -> (B, F)."""

    def __init__(self, n_rx: int) -> None:
        super().__init__()
        widths = (n_rx, *_ENCODER_CHANNELS)
        blocks = []
        for i in range(len(_ENCODER_CHANNELS)):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(widths[i], widths[i + 1], kernel_size=3, padding=1),
                    nn.BatchNorm2d(widths[i + 1]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, ceil_mode=True),
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)

    @property
    def out_features(self) -> int:
        return _ENCODER_CHANNELS[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.blocks(x)).flatten(1)


class CSICnnLstm(nn.Module):
    """Chunked CNN features through a BiLSTM; mean over steps -> logits."""

    def __init__(
        self,
        n_rx: int,
        n_time: int,
        n_subcarriers: int,
        n_classes: int,
        chunk_len: int = 25,
        hidden_size: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if n_rx < 1:
            raise ValueError(f"n_rx must be >= 1, got {n_rx}")
        if chunk_len < 1:
            raise ValueError(f"chunk_len must be >= 1, got {chunk_len}")
        if n_time < chunk_len:
            raise ValueError(
                f"n_time ({n_time}) must be >= chunk_len ({chunk_len}); "
                "at least one full chunk is required"
            )
        if n_subcarriers < 1:
            raise ValueError(f"n_subcarriers must be >= 1, got {n_subcarriers}")
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")
        self.n_rx = n_rx
        self.n_time = n_time
        self.n_subcarriers = n_subcarriers
        self.n_classes = n_classes
        self.chunk_len = chunk_len

        self.encoder = _ChunkEncoder(n_rx)
        self.lstm = nn.LSTM(
            input_size=self.encoder.out_features,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_rx, T, S) float -> (B, n_classes) logits."""
        if x.ndim != 4 or x.shape[1] != self.n_rx:
            raise ValueError(
                f"expected (B, {self.n_rx}, T, S), got shape {tuple(x.shape)}"
            )
        batch, n_rx, n_time, n_sub = x.shape
        n_chunks = n_time // self.chunk_len
        if n_chunks < 1:
            raise ValueError(
                f"T ({n_time}) shorter than chunk_len ({self.chunk_len})"
            )
        # Trim the remainder, then fold chunks into the batch dimension so
        # the encoder weights are shared across all steps.
        trimmed = x[:, :, : n_chunks * self.chunk_len, :]
        chunks = (
            trimmed.reshape(batch, n_rx, n_chunks, self.chunk_len, n_sub)
            .permute(0, 2, 1, 3, 4)
            .reshape(batch * n_chunks, n_rx, self.chunk_len, n_sub)
        )
        feats = self.encoder(chunks).reshape(batch, n_chunks, -1)
        out, _ = self.lstm(feats)
        pooled = out.mean(dim=1)
        return self.head(self.dropout(pooled))

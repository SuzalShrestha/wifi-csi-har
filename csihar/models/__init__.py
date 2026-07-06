"""Deep models for CSI HAR (Phase 4) and the factory the checkpoint loader uses.

``build_model`` is the single construction path: training, checkpoint
round-trips, and the Phase 5 real-time loader all go through it, so a model's
constructor signature is part of the checkpoint contract.
"""

from __future__ import annotations

from .cnn import CSICnn
from .cnn_lstm import CSICnnLstm

MODEL_NAMES: tuple[str, ...] = ("cnn", "cnn_lstm")


def build_model(
    name: str,
    n_rx: int,
    n_time: int,
    n_subcarriers: int,
    n_classes: int,
    chunk_len: int = 25,
):
    """Construct a model by registry name.

    ``chunk_len`` only applies to ``cnn_lstm`` and is ignored by ``cnn``.
    """
    if name == "cnn":
        return CSICnn(
            n_rx=n_rx,
            n_time=n_time,
            n_subcarriers=n_subcarriers,
            n_classes=n_classes,
        )
    if name == "cnn_lstm":
        return CSICnnLstm(
            n_rx=n_rx,
            n_time=n_time,
            n_subcarriers=n_subcarriers,
            n_classes=n_classes,
            chunk_len=chunk_len,
        )
    raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")


__all__ = ["CSICnn", "CSICnnLstm", "MODEL_NAMES", "build_model"]

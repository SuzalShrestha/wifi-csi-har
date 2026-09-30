"""On-device (ESP32-S3) inference budget for the Phase 5 stretch goal.

Answers the only question that decides whether int8 inference on the
receiver itself is worth attempting: does the model *fit*, and is the
arithmetic cheap enough to keep up with the hop?

Everything here is computed from the model's own layer shapes by running one
forward pass with hooks, so the numbers track the architecture rather than a
hand-maintained table. It deliberately does not depend on TensorFlow,
esp-tflite-micro, or any export toolchain: those are needed to *deploy*, not
to decide whether deploying is possible.

Budget model
------------
* **Weights** live in flash. int8 post-training quantization stores one byte
  per parameter plus per-channel scales; the scale overhead is negligible
  here and is ignored.
* **Activations** live in RAM, in what TFLite Micro calls the tensor arena.
  A sequential trunk keeps only the current op's input and output alive, so
  the arena floor is ``max(in + out)`` over ops. Real planners do somewhat
  better (buffer reuse) and real graphs somewhat worse (residuals keep extra
  tensors live); this model has no skip connections, so the estimate is
  tight. A safety factor covers scratch buffers and alignment.
* **Compute** is counted in multiply-accumulates. The S3 has no SIMD for
  int8 dot products, so the figure is a floor, not a prediction.

The single-receiver constraint
------------------------------
A receiver only holds *its own* CSI. The 3-receiver model cannot run on one
board without the boards forwarding CSI to each other, which reintroduces
the network dependency that on-device inference exists to remove. On-device
deployment therefore means the ``rx0`` (single-receiver) variant, and its
accuracy cost is exactly what the receiver-count ablation measures.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch import nn

from .models import MODEL_NAMES, build_model

# ESP32-S3-WROOM-1 N16R8. Internal SRAM is the figure that matters for the
# arena: PSRAM works but is markedly slower, and part of internal SRAM is
# already spoken for by the WiFi stack and FreeRTOS.
INTERNAL_SRAM_BYTES = 512 * 1024
PSRAM_BYTES = 8 * 1024 * 1024
FLASH_BYTES = 16 * 1024 * 1024

# Rough share of internal SRAM left after IDF, the WiFi stack, and the CSI
# ring buffer. Used only to label a verdict, never to compute one.
USABLE_SRAM_FRACTION = 0.5

# Covers scratch buffers, alignment padding, and planner slack.
ARENA_SAFETY_FACTOR = 1.3


@dataclass(frozen=True)
class LayerBudget:
    """One traced op: its output shape and the tensors live across it."""

    name: str
    kind: str
    out_shape: tuple[int, ...]
    in_bytes_int8: int
    out_bytes_int8: int
    macs: int

    @property
    def live_bytes_int8(self) -> int:
        return self.in_bytes_int8 + self.out_bytes_int8


@dataclass(frozen=True)
class DeviceBudget:
    """Whole-model deployment budget, all sizes in bytes."""

    model: str
    input_shape: tuple[int, ...]
    n_params: int
    weights_float32: int
    weights_int8: int
    arena_int8: int
    arena_float32: int
    macs: int
    layers: tuple[LayerBudget, ...]

    @property
    def arena_with_safety(self) -> int:
        return int(self.arena_int8 * ARENA_SAFETY_FACTOR)

    @property
    def fits_internal_sram(self) -> bool:
        return self.arena_with_safety <= INTERNAL_SRAM_BYTES * USABLE_SRAM_FRACTION

    @property
    def fits_flash(self) -> bool:
        return self.weights_int8 <= FLASH_BYTES


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _macs(module: nn.Module, inp: torch.Tensor, out: torch.Tensor) -> int:
    """Multiply-accumulates of one op; 0 for shape-only and elementwise ops."""
    if isinstance(module, nn.Conv2d):
        # One output element costs (in_channels/groups * kh * kw) MACs.
        kh, kw = module.kernel_size
        per_out = (module.in_channels // module.groups) * kh * kw
        return _numel(tuple(out.shape)) * per_out
    if isinstance(module, nn.Linear):
        return _numel(tuple(out.shape)) * module.in_features
    if isinstance(module, nn.LSTM):
        # 4 gates, each a matmul over input and over hidden state, per step,
        # per direction.
        n_dirs = 2 if module.bidirectional else 1
        steps = int(inp.shape[1]) if module.batch_first else int(inp.shape[0])
        h = module.hidden_size
        per_step = 4 * h * (module.input_size + h)
        return steps * per_step * n_dirs
    return 0


_COUNTED = (nn.Conv2d, nn.Linear, nn.LSTM, nn.MaxPool2d, nn.AdaptiveAvgPool2d)


def profile_model(
    name: str,
    *,
    n_rx: int,
    n_time: int,
    n_subcarriers: int,
    n_classes: int,
) -> DeviceBudget:
    """Trace one forward pass and total up the deployment budget."""
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")
    model = build_model(
        name,
        n_rx=n_rx,
        n_time=n_time,
        n_subcarriers=n_subcarriers,
        n_classes=n_classes,
    ).eval()

    layers: list[LayerBudget] = []
    handles = []

    def hook(mod: nn.Module, args, out) -> None:
        inp = args[0] if isinstance(args, tuple) else args
        if not isinstance(inp, torch.Tensor):
            return
        out_t = out[0] if isinstance(out, tuple) else out
        if not isinstance(out_t, torch.Tensor):
            return
        layers.append(
            LayerBudget(
                name=mod._profile_name,
                kind=type(mod).__name__,
                out_shape=tuple(int(d) for d in out_t.shape),
                in_bytes_int8=_numel(tuple(inp.shape)),
                out_bytes_int8=_numel(tuple(out_t.shape)),
                macs=_macs(mod, inp, out_t),
            )
        )

    for mod_name, mod in model.named_modules():
        if isinstance(mod, _COUNTED):
            mod._profile_name = mod_name or type(mod).__name__
            handles.append(mod.register_forward_hook(hook))

    x = torch.zeros(1, n_rx, n_time, n_subcarriers)
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()

    n_params = sum(p.numel() for p in model.parameters())
    arena_int8 = max((lb.live_bytes_int8 for lb in layers), default=0)
    return DeviceBudget(
        model=name,
        input_shape=(n_rx, n_time, n_subcarriers),
        n_params=n_params,
        weights_float32=n_params * 4,
        weights_int8=n_params,
        arena_int8=arena_int8,
        arena_float32=arena_int8 * 4,
        macs=sum(lb.macs for lb in layers),
        layers=tuple(layers),
    )


def _kb(n: int) -> str:
    return f"{n / 1024:.1f} KB"


def format_budget(b: DeviceBudget, *, per_layer: bool = False) -> str:
    """Human-readable report; the CLI prints exactly this."""
    lines = [
        f"model                {b.model}  input {b.input_shape}",
        f"parameters           {b.n_params:,}",
        f"weights   float32    {_kb(b.weights_float32)}",
        f"weights   int8       {_kb(b.weights_int8)}",
        f"arena     float32    {_kb(b.arena_float32)}",
        f"arena     int8       {_kb(b.arena_int8)}"
        f"  (+{int((ARENA_SAFETY_FACTOR - 1) * 100)}% safety"
        f" -> {_kb(b.arena_with_safety)})",
        f"compute              {b.macs / 1e6:.1f} MMAC per window",
        "",
        "ESP32-S3-WROOM-1 N16R8",
        f"  flash 16 MB        weights fit: {'YES' if b.fits_flash else 'NO'}",
        f"  internal SRAM 512 KB, assume {int(USABLE_SRAM_FRACTION * 100)}%"
        f" free ({_kb(int(INTERNAL_SRAM_BYTES * USABLE_SRAM_FRACTION))})",
        f"  arena fits internal SRAM: "
        f"{'YES' if b.fits_internal_sram else 'NO - needs PSRAM'}",
        f"  PSRAM 8 MB         arena fits: "
        f"{'YES' if b.arena_with_safety <= PSRAM_BYTES else 'NO'}",
    ]
    if per_layer:
        lines += ["", f"{'layer':<28}{'kind':<18}{'out shape':<22}"
                      f"{'live int8':>10}{'MMAC':>9}"]
        lines.append("-" * 87)
        for lb in b.layers:
            lines.append(
                f"{lb.name:<28}{lb.kind:<18}{str(lb.out_shape):<22}"
                f"{_kb(lb.live_bytes_int8):>10}{lb.macs / 1e6:>9.2f}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m csihar.ondevice",
        description=(
            "Memory and compute budget for running a trained model on an "
            "ESP32-S3 receiver (Phase 5 stretch goal). Note that a receiver "
            "only has its own CSI, so on-device deployment means --n-rx 1."
        ),
    )
    parser.add_argument("--model", choices=MODEL_NAMES, default="cnn")
    parser.add_argument("--n-rx", type=int, default=1,
                        help="1 for on-device (a board has only its own CSI)")
    parser.add_argument("--n-time", type=int, default=300, help="3 s at 100 Hz")
    parser.add_argument("--n-subcarriers", type=int, default=52)
    parser.add_argument("--n-classes", type=int, default=6)
    parser.add_argument("--per-layer", action="store_true")
    args = parser.parse_args(argv)

    budget = profile_model(
        args.model,
        n_rx=args.n_rx,
        n_time=args.n_time,
        n_subcarriers=args.n_subcarriers,
        n_classes=args.n_classes,
    )
    print(format_budget(budget, per_layer=args.per_layer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

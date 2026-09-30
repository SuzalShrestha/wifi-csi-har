# On-device inference on the ESP32-S3 — feasibility study

**Date:** 2026-08-28 · **Status:** desk analysis complete, not deployed
**Reproduce:** `.venv/bin/python -m csihar.ondevice --n-rx 1 --per-layer`

IMPLEMENTATION_PLAN.md Phase 5 lists int8 on-device inference as a stretch
goal with high defense value. This is the analysis that decides whether it
is worth attempting. Every number below is computed from the model's own
traced layer shapes by [csihar/ondevice.py](../csihar/ondevice.py), so it
tracks the architecture rather than a hand-maintained table.

## Verdict

**Feasible, but not with the current architecture running from internal
SRAM.** Three findings, in order of how much they constrain the design.

### 1. Weights are not the constraint

| | float32 | int8 |
|---|---|---|
| CNN (241,734 params) | 944 KB | **236 KB** |
| CNN-LSTM (219,206 params) | 856 KB | **214 KB** |

Against 16 MB of flash, either model fits with three orders of magnitude to
spare. Quantization is worth doing for speed, not for fit.

### 2. The tensor arena is the constraint, and it lives in the first conv block

Peak live working set, 3 s window, single receiver (int8):

| layer | output shape | live |
|---|---|---|
| `features.0.0` Conv2d | (1, 32, 300, 52) | 503 KB |
| **`features.0.3` MaxPool2d** | (1, 32, 150, 26) | **609 KB** ← peak |
| `features.1.0` Conv2d | (1, 64, 150, 26) | 366 KB |
| `features.2.0` Conv2d | (1, 128, 75, 13) | 183 KB |
| `features.3.0` Conv2d | (1, 128, 38, 7) | 67 KB |
| `head` Linear | (1, 6) | 0.1 KB |

609 KB, or **792 KB with a 30% safety factor**, against 512 KB of *total*
internal SRAM — of which the WiFi stack, FreeRTOS, and the CSI ring buffer
have already taken roughly half.

The peak is entirely the first convolution's full-resolution 32-channel
feature map. Nothing later in the trunk comes close, because every
subsequent block has already been pooled.

Nothing in the current grid fits internal SRAM (target: ~256 KB free):

| model | window | arena +30% | MMAC | fits internal SRAM |
|---|---|---|---|---|
| cnn | 3.0 s | 792 KB | 187.5 | no |
| cnn | 1.5 s | 396 KB | 94.2 | no |
| cnn | 1.0 s | 264 KB | 62.8 | no (just barely) |
| cnn_lstm | 3.0 s | 799 KB | 81.6 | no |
| cnn_lstm | 1.0 s | 266 KB | 27.2 | no (just barely) |

It fits the 8 MB PSRAM with enormous margin — but PSRAM is markedly slower
than internal SRAM, and the arena is exactly the memory the inner loop
touches most.

**The remedy is a strided front end.** Because the peak is the first feature
map, a stride-2 first convolution (or a 2× pool before it) cuts the arena
and the compute by roughly 4× each — approximately 200 KB and 47 MMAC for
the full 3 s window, which fits internal SRAM comfortably. This costs
accuracy that has not been measured, so it is a proposal, not a decision;
it would need its own row in the results table.

### 3. On-device means single-receiver — and that is the real cost

A receiver only holds **its own** CSI. Running the 3-receiver model on one
board would require the boards to forward CSI to each other, which
reintroduces exactly the network dependency that on-device inference exists
to remove.

So on-device deployment is the `rx0` variant, and its accuracy cost is
precisely what the receiver-count ablation measures. Memory is not what
makes this expensive — receivers are input channels, so `n_rx` changes the
parameter count by under 1% — **the cost is entirely the loss of spatial
diversity.**

This is the number that decides the stretch goal, and it does not exist yet:
it needs the receiver-count ablation on real multi-subject data.

## Compute

187.5 MMAC per window for the 3 s single-receiver CNN. The ESP32-S3 at
240 MHz with ESP-NN's int8 SIMD kernels plausibly sustains a few MACs per
cycle, putting one inference somewhere in the several-hundred-millisecond
range against a 1500 ms hop — feasible in principle, tight in practice, and
worse if the arena is in PSRAM. This is an order-of-magnitude estimate from
the MAC count, **not a measurement**; the S3's actual int8 throughput has
not been benchmarked here.

## What remains, and what it needs

| Step | Needs hardware? |
|---|---|
| Budget analysis (this document) | no — **done** |
| Choose/validate a strided front end, retrain, measure the accuracy cost | no |
| Receiver-count ablation on real data (the deciding number) | data |
| Export to int8 TFLite (post-training quantization, representative set) | no — needs TF installed |
| esp-tflite-micro integration + flash + on-board latency measurement | **yes** |

## Recommendation

Do not start this until the full dataset exists and the receiver-count
ablation has run. If the single-receiver macro-F1 is within a few points of
three receivers, the stretch goal is worth the firmware work and the report
gets a genuinely distinctive result. If the drop is large, on-device
inference is a demo that undercuts the project's own central claim about
multi-receiver diversity, and the effort belongs elsewhere.

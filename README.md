# WiFi CSI Human Activity Recognition (ESP32-S3)

Major project (EX 707), Tribhuvan University, Pashchimanchal Campus.
Contactless HAR from WiFi Channel State Information using 3× ESP32-S3 N16R8
receivers + a commodity router, classified with deep neural networks
(CNN / CNN-LSTM). See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the
full 8-month plan and [docs/collection_protocol.md](docs/collection_protocol.md)
for the data protocol.

## Layout

| Path | What |
|---|---|
| `firmware/` | esp-csi `csi_recv_router` setup + flash guide |
| `csihar/` | host pipeline: parser, collector, preprocessing, viewer |
| `csihar/preprocessing/` | subcarrier selection, Hampel/detrend/lowpass, windowing, normalization |
| `tests/` | pytest suite |
| `datasets/` | raw sessions (git-ignored) + datasheet |
| `docs/` | protocol, figures, decision log |

## Quick start (no hardware needed)

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest                 # 29 tests
.venv/bin/python -m csihar.view --simulate walking   # synthetic heatmap
```

## With hardware

1. Flash the boards: see [firmware/README.md](firmware/README.md)
2. Live heatmap: `python -m csihar.view --live /dev/cu.usbmodem<N>`
3. Record a session: see the command template in the protocol doc

## Pipeline

```
ESP32-S3 (esp-csi, 100 Hz ping → CSI_DATA over USB-serial 921600)
  → csihar.parser    (64 complex LLTF subcarriers per packet)
  → csihar.collector (3 threads, host timestamps, Parquet per receiver)
  → preprocessing    (52 usable subcarriers → Hampel → detrend/lowpass
                      → 3 s windows @ 50% overlap → window-level RX alignment
                      → train-only standardization)
  → models           (Phase 4: SVM baseline, CNN, CNN-LSTM)
```

# WiFi CSI HAR — ESP32-S3 (TU Major Project, EX 707)

Contactless human activity recognition from WiFi Channel State Information.
3× ESP32-S3 N16R8 receivers + commodity router (TX). 5 classes: walking,
sitting, standing, lying, falling (+ background). Team of 4, defense
~March–April 2027 (Chaitra). Target ≥85% accuracy on random/cross-session
splits. **Read [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) before
planning any new work** — it defines the 6 phases, milestones, and risk
register. Current status: Phase 1 host pipeline done; hardware bring-up and
Phase 2 (preprocessing pilot) next.

## Commands

```bash
.venv/bin/python -m pytest                          # test suite — keep green
.venv/bin/pip install -e ".[dev]"                   # after dependency changes
.venv/bin/python -m csihar.view --simulate walking  # pipeline demo, no hardware
.venv/bin/python -m csihar.view --live <port>       # live heatmap from a board
.venv/bin/python -m csihar.collector --help         # record a session
```

Python 3.14 venv at `.venv/`. No GPU on this machine — heavy training happens
on Colab/Kaggle; keep model code runnable on CPU for smoke tests.

## Repo map

- `csihar/parser.py` — CSI_DATA line → `CsiFrame`. Format is locked to
  esp-csi `csi_recv_router` (see below). Don't change without re-checking
  firmware source.
- `csihar/collector.py` — threaded 3-receiver serial capture → Parquet.
- `csihar/preprocessing/` — pure functions: subcarriers, filters, windowing,
  normalize.
- `csihar/simulate.py` — synthetic CSI in byte-exact firmware format; use it
  to develop/test anything downstream without hardware.
- `csihar/storage.py` — Parquet + metadata.json session layout.
- `firmware/` — cloned espressif/esp-csi (git-ignored) + flash guide.
- `docs/collection_protocol.md` — data collection rules; fill blanks, don't
  drift from it silently.

## Hard-won facts — do not rediscover these

- **Firmware is espressif/esp-csi (`csi_recv_router` example), NOT
  ESP32-CSI-Tool** (stale, IDF 4.3). ESP32-S3 emits 24 metadata columns then
  `"[...]"` with `len` int8 values.
- **CSI bytes are (imaginary, real) pairs — imag first** (ESP-IDF docs).
  Swapping them corrupts all amplitudes. Parser handles this; tests pin it.
- First 64 complex values = LLTF (20 MHz). **52 usable subcarriers**:
  buffer indices 1–26 and 38–63; DC (0) and guard band (27–37) are null.
  Theory not yet confirmed on real hardware — run
  `preprocessing.detect_null_subcarriers` on the first real capture and
  update `usable_lltf_indices` if they disagree.
- **Serial must be 921600 baud** (115200 drops lines at 100 pkt/s).
- **Amplitude only.** Single antenna → raw phase is CFO/SFO-corrupted;
  multi-antenna phase sanitization is impossible on this hardware. Don't
  build phase features into the main pipeline.
- Receivers have independent clocks: align at **window level** via host
  timestamps, never per-packet.
- NumPy 2.x: `np.fromstring` is gone (already hit this once).

## Non-negotiable evaluation rules (defense depends on these)

1. Every model result reports **three splits**: random-window,
   cross-session, cross-subject (leave-one-subject-out). Never quote a
   random-split number alone — temporal leakage inflates it.
2. Scalers/augmentation statistics fit on **train split only**
   (`fit_scaler` enforces the shape; you enforce the discipline).
3. Falling is safety-critical: always report its recall separately.
4. Every experiment: fixed seed, config committed, results CSV under
   `experiments/`. Figures regenerate from scripts — no hand-made plots.

## Conventions

- Immutable data: frozen dataclasses, pure functions returning new arrays
  (repo-wide rule; preprocessing tests assert non-mutation).
- Many small files; new pipeline stages get unit tests in the same PR.
- Never commit `datasets/` contents, Parquet files, or `firmware/esp-csi/`
  (all git-ignored). Dataset sharing goes via Drive; metadata.json sidecars
  make sessions self-describing.
- Commit style: `<type>: <description>` (feat/fix/refactor/docs/test/chore).

## Skills

- `/flash-firmware` — set up ESP-IDF and flash a receiver board
- `/collect-session` — record a labeled data session correctly
- `/validate-session` — post-session QA before accepting data
- `/train-model` — Phase 4 training/evaluation workflow and guardrails

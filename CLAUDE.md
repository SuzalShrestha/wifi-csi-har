# WiFi CSI HAR — ESP32-S3 (TU Major Project, EX 707)

Contactless human activity recognition from WiFi Channel State Information.
3× ESP32-S3 N16R8 receivers + commodity router (TX). 5 classes: walking,
sitting, standing, lying, falling (+ background). Team of 4, defense
~March–April 2027 (Chaitra). Target ≥85% accuracy on random/cross-session
splits. **Read [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) before
planning any new work** — it defines the 6 phases, milestones, and risk
register. Current status: ALL host-side software (Phases 1–6) is built and
tested on simulated data — parser -> collector -> preprocessing ->
dataset/splits -> baseline + CNN/CNN-LSTM training -> ablation runner
(csihar/experiments.py) -> realtime engine -> web dashboard
(csihar/dashboard.py) -> figures pipeline (csihar/figures.py) + LaTeX report
skeleton (report/). Everything remaining is hardware/data work: flash boards
(/flash-firmware), confirm subcarrier nulls on real captures, collect the
pilot dataset (/collect-session), rerun training/ablations on real data,
measure real-time latency, then write the report chapters.

## Commands

```bash
.venv/bin/python -m pytest                          # test suite — keep green
.venv/bin/pip install -e ".[dev,ml,demo]"           # after dependency changes
.venv/bin/python -m csihar.view --simulate walking  # pipeline demo, no hardware
.venv/bin/python -m csihar.view --live <port>       # live heatmap from a board
.venv/bin/python -m csihar.collector --help         # record a session
.venv/bin/python -m csihar.experiments --help       # Phase 4 ablations
.venv/bin/python -m csihar.dashboard --help         # Phase 5 demo dashboard
make figures                                        # regenerate report figures
make report                                         # build LaTeX report (needs TeX)
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
- `docs/research_review.md` — 2026-07-16 literature audit: what was fixed
  (regime-matched early-stopping val, class-weighted loss, fall-alert fast
  path) and the deferred gaps with their triggers. Check it before adding
  augmentation/denoising or touching evaluation code.

## Hard-won facts — do not rediscover these

- **Firmware is espressif/esp-csi (`csi_recv_router` example), NOT
  ESP32-CSI-Tool** (stale, IDF 4.3). ESP32-S3 emits 24 metadata columns then
  `"[...]"` with `len` int8 values.
- **CSI bytes are (imaginary, real) pairs — imag first** (ESP-IDF docs).
  Swapping them corrupts all amplitudes. Parser handles this; tests pin it.
- First 64 complex values = LLTF (20 MHz). **52 usable subcarriers**:
  buffer indices 1–26 and 38–63; DC (0) and guard band (27–37) are null.
  **Confirmed on real hardware 2026-07-13** — `detect_null_subcarriers` on a
  2500-frame live capture returned exactly the theoretical nulls.
- **CSI needs sustained UDP downlink traffic** (confirmed 2026-07-13): with
  only the firmware's ping, the router sends replies/beacons at DSSS/CCK
  rates which carry no OFDM LTF → <1 Hz CSI. The host must flood each
  receiver's IP with ~100 pkt/s of 200-byte UDP (`csihar/traffic.py`,
  `--traffic` flag on collector/view) → steady 100 Hz. Not an IDF version
  issue (identical on v5.3.2 and v5.4.4).
- Firmware needs `esp_wifi_set_ps(WIFI_PS_NONE)` and a **custom console
  UART** (USB-JTAG console sends CSI out the wrong USB port; baud is only
  configurable in custom mode). Both captured in
  `firmware/patches/csihar.patch` — apply after any esp-csi re-clone.
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

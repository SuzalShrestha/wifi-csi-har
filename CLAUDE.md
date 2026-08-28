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
.venv/bin/python -m csihar.preflight --help         # verify the rig BEFORE recording
.venv/bin/python -m csihar.collector --help         # record a session
.venv/bin/python -m csihar.datasheet datasets/raw    # collection progress vs M3 targets
.venv/bin/python -m csihar.experiments --help       # Phase 4 ablations
.venv/bin/python -m csihar.dashboard --help         # Phase 5 demo dashboard
.venv/bin/python -m csihar.latency --help           # pipeline latency benchmark
make figures                                        # regenerate report figures
make report                                         # build LaTeX report (needs TeX)
```

Python 3.14 venv at `.venv/`. No CUDA GPU on this machine, but **Apple MPS
works and is ~2.5x faster than CPU** (`--device auto` picks cuda > mps > cpu).
Heavy training goes to Colab via `notebooks/train_colab.ipynb`; keep model
code runnable on CPU for smoke tests.

**Results are not reproducible across devices.** The same seed on CPU and MPS
gave test accuracy 0.903 vs 0.929 on identical data; the same CNN on a Colab
T4 gave 0.937. The *resolved* device (not `"auto"`) is written into the
results CSV's config column — pin it when reporting a number, and never
compare rows trained on different devices.

## Repo map

- `csihar/parser.py` — CSI_DATA line → `CsiFrame`. Format is locked to
  esp-csi `csi_recv_router` (see below). Don't change without re-checking
  firmware source.
- `csihar/collector.py` — threaded 3-receiver serial capture → Parquet.
- `csihar/preflight.py` — 10 s live check before a session: every receiver
  present, at rate, on one channel. Run it first; it is far cheaper than
  discovering a dead board after a 25-minute recording.
- `csihar/session_script.py` — guided multi-segment session: one continuous
  recording, `labels.json` sidecar of host-clock label ranges. Use this for
  pilot/full collection rather than one file per activity.
- `csihar/traffic.py` — UDP downlink generator; `downlink_traffic()` context
  manager and `add_traffic_argument()` are the shared wiring every live
  entry point uses.
- `csihar/preprocessing/` — pure functions: subcarriers, filters, windowing,
  normalize.
- `csihar/simulate.py` — synthetic CSI in byte-exact firmware format; use it
  to develop/test anything downstream without hardware.
- `csihar/storage.py` — Parquet + metadata.json session layout. Set
  `exclude_from_dataset=True` on rig-test captures; `assemble_dataset` and
  `datasheet` both honour it. A note in `notes` is not enough — nothing reads
  it, and an uncontrolled-room capture reached a reported result that way.
- `csihar/datasheet.py` — running collection report: seconds and estimated
  windows per class/subject/session against the M3 targets.
- `firmware/` — cloned espressif/esp-csi (git-ignored) + flash guide.
- `csihar/latency.py` — per-stage latency benchmark (parse/ingest/window/
  inference) over a simulated or replayed session; runs against an untrained
  model since latency is weight-independent.
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
  receiver's IP with ~100 pkt/s of 200-byte UDP → steady 100 Hz. Not an IDF
  version issue (identical on v5.3.2 and v5.4.4). **Every entry point that
  reads live from the boards takes `--traffic IP` (repeatable) and routes it
  through `traffic.downlink_traffic`** — collector, session_script, view,
  realtime, dashboard. `tests/test_traffic.py` pins this for all five;
  session_script/realtime/dashboard shipped without it once and would have
  silently recorded worthless sessions.
- Firmware needs `esp_wifi_set_ps(WIFI_PS_NONE)` and a **custom console
  UART** (USB-JTAG console sends CSI out the wrong USB port; baud is only
  configurable in custom mode). Both captured in
  `firmware/patches/csihar.patch` — apply after any esp-csi re-clone.
- **Serial must be 921600 baud** (115200 drops lines at 100 pkt/s).
- **Plug into the devkit's UART USB-C port, not the native USB port.** Both
  enumerate, and the native USB port emits CSI too (115200 secondary
  console), so a board on the wrong port looks like it is working — this
  cost a bring-up session. Tell them apart on macOS: the UART bridge is
  `USB Single Serial` (CH343, `/dev/cu.usbmodem5XXXXXXXXXX`), the native
  port is `USB JTAG_serial debug unit` (VID 0x303a PID 0x1001). Only the
  UART path is validated at 100 Hz.
- **Don't reset a board with DTR/RTS over the native USB port** — the
  esptool-style pulse drops ESP32-S3 into `waiting for download`, which
  reads exactly like an unflashed board. Verified boards look blank this way.
- **Verified end-to-end 2026-08-27** (3 boards, SSID `shrestha`, ch 6 BW20):
  100.2-100.6 Hz per receiver, 0 dropped sequence numbers, 0 malformed lines
  over 20 s; 2500 live frames parsed with 0 errors and
  `detect_null_subcarriers` again returning exactly DC + bins 27-37. Port /
  IP / MAC table is in `docs/collection_protocol.md`.
- **Pipeline latency is structural, not computational.** Measured 2026-08-27
  (`csihar/latency.py`, 3 receivers, 3 s window / 1.5 s hop, replaying the
  real bring-up capture): tick compute p95 ~23 ms against a 1500 ms hop
  budget — 65x headroom. Window construction (hampel + detrend + resample) is
  ~85% of that; inference is ~3 ms; parse and ingest are ~0.02 ms per frame.
  End-to-end delay is dominated by the 3 s of buffering a window needs. To cut
  latency, shorten `window_s` — optimizing code will not move it.
- **The streaming engine scores ~5 points below the offline pipeline on the
  same windows, and that gap is structural.** Measured 2026-08-28 replaying
  `20260827_232655_sujal_scripted`: offline 0.9177, realtime raw 0.8671, 96%
  window-for-window prediction agreement. Cause: `detrend_moving_mean` is a
  **centered** 101-tap kernel, so offline every interior sample sees 0.5 s of
  *future* data. `_slice_window` grabs 0.5 s of lead-in but cannot grab
  trailing context — that is the future. Error is ~0.0001 mid-window and
  0.91 in the last 0.5 s. Closing it would cost 0.5 s of added latency; we
  have the headroom, but it buys ~5 points, so it is a deliberate trade, not
  a bug. Do not "fix" `_slice_window` expecting a large win.
- **Smoothed accuracy is LOWER than raw — report raw.** End-to-end 0.8025
  smoothed vs 0.8671 raw. Majority voting helps only against independent
  noise; these errors cluster. The smoother earns its place on display
  stability (29 label changes -> 19 over 20 min, against 3 real ones), not
  accuracy. The dashboard streams both labels; quote the raw number.
- **A confidence threshold must abstain, not vote.** `smooth_predictions`
  used to map low-confidence windows to the literal label `"unknown"`, which
  then competed in the majority vote — two unsure windows could veto three
  confident ones. That made the smoother strictly dominated: identical
  flicker to raw and 6 points worse. Low-confidence entries now drop out of
  the vote; `"unknown"` is returned only when nothing recent is confident.
- **Amplitude only.** Single antenna → raw phase is CFO/SFO-corrupted;
  multi-antenna phase sanitization is impossible on this hardware. Don't
  build phase features into the main pipeline.
- Receivers have independent clocks: align at **window level** via host
  timestamps, never per-packet.
- **The results `notes` column carries both structured tokens and free text**,
  joined with `"; "` — `train` appends the degenerate-split coverage warning
  to whatever notes the caller set. `figures.parse_notes` therefore splits on
  `;` as well as whitespace; splitting on whitespace alone left the semicolon
  glued to the value (`variant=25Hz;`), which made `make figures` raise on
  window/rate ablation rows and silently split the receivers series in two.
  Anything else that parses `notes` must do the same.
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
   Seed alone does not pin a run: the device does too (see above).
5. **A high accuracy with a macro-F1 near 1/n_classes is a degenerate
   split, not a result** — Colab reported cross-session 0.9749 accuracy /
   0.1645 macro-F1 on a test session holding only `background`. Both
   `baseline` and `train` now call `split_coverage_note` and write it into
   the CSV's `notes`; a cross-* row with an empty `notes` predates that fix.
   The checkpoint from such a run is still usable — only the metric is void.
6. A split whose test set is missing classes the train set has is
   **degenerate, not bad** — its accuracy is uninterpretable.
   `dataset.split_coverage_note` detects this, `baseline` warns and writes
   the reason into the results CSV. Cross-session needs activities in >= 2
   sessions; cross-subject needs >= 2 subjects.

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

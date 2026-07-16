# Research review — approach audit vs. WiFi CSI HAR literature

**Date:** 2026-07-16 · **Scope:** full host-side pipeline (Phases 1–6 code) reviewed
against the CSI HAR literature the project builds on (SenseFi benchmark, Moshiri
et al. CSI-HAR, ESP32 HAR feasibility papers, fall-detection work such as
WiFall/RT-Fall, and the ACM Computing Surveys 2025 review). Findings were either
**fixed in code** (with tests) or **deferred with a trigger** for when real data
arrives.

---

## 1. What the approach gets right (validated, keep as-is)

| Decision | Literature backing |
|---|---|
| Amplitude-only features | Single-antenna ESP32-S3 phase is CFO/SFO-corrupted; multi-antenna sanitization impossible. All credible ESP32 HAR work (incl. Moshiri et al.) is amplitude-only. |
| 100 Hz packet rate, 3 s windows, 50 % overlap | Human-motion Doppler < ~60 Hz; window/rate choices sit inside the ranges SenseFi-style benchmarks use, and both are ablated (1–3 s, 25–100 Hz). |
| Hampel → moving-mean detrend | Hampel is the standard first stage for impulsive RF/AGC spikes; mean-subtraction removes the static channel component (what CARM-style pipelines achieve with high-pass). |
| 52 usable LLTF subcarriers (pilots kept) | Confirmed empirically on real hardware 2026-07-13; keeping pilots is standard for amplitude pipelines. |
| Three-split evaluation (random / cross-session / cross-subject) | The surveys' core criticism of the field is random-split-only reporting; this repo enforces all three in code. |
| Window-level (not packet-level) receiver alignment | Receivers have independent clocks; packet-level alignment is impossible without shared time sync. |
| Classical baseline gate (SVM/RF must be beaten) | SenseFi's central finding: simple baselines are competitive; a DL model that can't beat them signals a broken pipeline. |

---

## 2. Gaps and errors found — FIXED in this pass

### 2.1 Early stopping validated on temporally-leaky windows (worst finding)
`train.py` carved validation from train with a *random stratified window split*
for **every** split regime. Overlapping windows from the same activity bout landed
in both fit and val, so val macro-F1 tracked memorization — exactly the leakage
the project's own plan (§Phase 4: "early-stop on validation from the same split
regime being tested") forbids. For cross-session/cross-subject runs the patience
logic could select the most-overfit epoch.
**Fix:** `_grouped_val_split` in [train.py](../csihar/train.py) — for cross-*
splits, validation now holds out whole sessions (falls back to stratified with a
loud warning only when train has a single session). Random split keeps the
stratified val (regime-matched by definition).
*Deliberate compromise:* for LOSO runs the val shift is session-level, not
subject-level — holding out a whole validation subject is too costly with 4–10
subjects. Session-held-out val removes the temporal leakage, which is the
dominant error; note this in the report's methodology chapter.

### 2.2 No class weighting despite a scarce safety-critical class
Plan promised class weighting/oversampling for falls; `CrossEntropyLoss()` was
unweighted. Falls will be the scarcest class (mattress protocol) and are the class
whose recall the defense reports separately.
**Fix:** inverse-frequency `_class_weights` on the fit split, default-on via
`TrainConfig.class_weighted` ([train.py](../csihar/train.py)).

### 2.3 Fall alert could never fire in the live demo
Dashboard raised the fall banner only when the *smoothed* label was `falling`.
A fall is transient — it occupies 1–2 windows, so it reliably **loses** the
5-vote majority; the safety-critical alert was structurally suppressed by the
very smoothing added to stabilize the other classes (a known failure mode in
fall-detection literature — transients must bypass temporal smoothing).
**Fix:** `Prediction.fall_alert` fast path in [realtime.py](../csihar/realtime.py):
a confident raw `falling` window (≥ smoother `min_confidence`) alerts immediately,
independent of the majority vote; dashboard now forwards it.

### 2.4 Silent train/serve window-length mismatch
The CNN ends in adaptive pooling, so it *accepts any T*. The realtime CLI always
used the default 3 s window; a checkpoint trained at another window length would
be scored on a shifted distribution with no error.
**Fix:** realtime derives `window_s` from the checkpoint's `n_time`, and
`predict()` now rejects a T that differs from the trained one
([realtime.py](../csihar/realtime.py), [models/inference.py](../csihar/models/inference.py)).

### 2.5 Non-deterministic resample grid length at epoch timestamps
`np.arange(t_start, t_end − 1e-9, 1/fs)` over host timestamps (~1.7 × 10⁹ s) can
yield 299 or 301 samples from float accumulation; one odd-sized window breaks
`np.stack` at dataset assembly, potentially hours into a collection run.
**Fix:** grid length computed as `round((t_end − t_start) · fs)` exactly
([windowing.py](../csihar/preprocessing/windowing.py)).

### 2.6 Unbounded prediction history in the realtime engine
`RealtimeEngine._history` grew forever (memory leak over a long demo); the
smoother only ever reads the last `vote_k` entries. **Fix:** pruned each tick.

### 2.7 Even detrend window silently misaligned the trend
`detrend_moving_mean` accepted even windows, shifting the trend half a sample.
**Fix:** odd-window validation (same rule Hampel already had).

### 2.8 No cross-environment split despite a planned second-room transfer test
Plan §Phase 3 requires a second-room test set; the dataset carries an
`environments` tag but had no split helper. **Fix:** `split_cross_environment`
in [dataset.py](../csihar/dataset.py) with the same no-straddle guarantee as the
other group splits.

All fixes are covered by new unit tests; full suite: **165 passed**.

---

## 3. Deferred gaps — documented, with triggers

| Gap | Why deferred | Trigger to act |
|---|---|---|
| Data augmentation (time-warp, subcarrier dropout, amplitude jitter) | Untunable without real cross-subject numbers | After first real-data LOSO run, if falling recall or cross-subject macro-F1 lags |
| MCS mixed-rate window filtering (risk register) | Need real MCS distributions to pick a policy; QA currently checks rate/gaps/RSSI/nulls only | First real multi-hour session: inspect per-packet MCS spread, add a QA check + window filter if mixed |
| Detection-delay metric (activity-onset latency, plan §Phase 5) | Needs real labeled onsets | Measure during pilot-dataset real-time runs; report alongside latency |
| Low-pass / DWT denoising stage (plan §Phase 2 mentions it; `filters.lowpass` exists but is not in the pipeline) | Whether it helps depends on real noise; CNNs often absorb it | Run as a preprocessing ablation on the pilot dataset before freezing preprocessing (M2) |
| Streaming detrend edge skew | Offline detrend sees future context; live windows can't at the right edge. Irreducible; magnitude unknown | Quantify on pilot data (offline vs streaming features on same capture); mention in report failure analysis |
| `band_energy_0_2hz` includes the DC bin, overlapping the `mean` feature | Harmless for RF/SVM; changing it churns pinned tests for no expected gain | Only if baseline feature importance analysis is done for the report |
| `preprocessing/normalize.py` (`fit_scaler`) is not on the training path (train.py's per-(rx, subcarrier) `norm_fit` is) | Duplicate scaler; candidate for deletion | Next cleanup pass — delete or route train.py through it, not both |
| Cross-environment split not exposed in the train CLI | No second-room data exists yet | When room-B sessions are collected, wire `--split cross-environment` |

## 4. Evaluation-rule addendum (extends CLAUDE.md rules)

- Early stopping for cross-session / cross-subject runs uses **session-held-out
  validation** (automatic in `train_model`); a stratified fallback warning in the
  logs means the train side had a single session — treat that run's early
  stopping as optimistic.
- Fall alerts in demos come from `Prediction.fall_alert` (raw fast path), never
  from the smoothed label alone.

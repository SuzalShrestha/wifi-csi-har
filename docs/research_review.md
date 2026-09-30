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

---

# Audit 2 — 2026-09-30: project state, direction, performance

**Scope:** repo state, full host-side code path, schedule vs.
IMPLEMENTATION_PLAN, and a literature check of two design choices that the
pilot-1 results put in question. Code fixes are on branch
`audit/2026-09-30`.

## A. Project state

- `main` was behind by PR #3 (8 commits, open since 2026-08-28) and by the
  local-only branch `docs/status-audit-and-report-chapters`. Both rewrote
  report chapters 1–4 in parallel. Merged on the audit branch; PR #3's
  drafts are a structural superset on every chapter, so conflicts went to
  them. `AGENTS.md` and `.agents/` were stale copies of `CLAUDE.md` and
  `.claude/skills`, so they were dropped. The docs branch still holds them.
- **No commits for a month, and no second session.** Collection is ~5% of
  M3 (1 subject, 1 day, 0 falls). IMPLEMENTATION_PLAN puts full collection
  in Bhadra/Ashwin, a window that is now closing.
- Pilot-1 raw Parquet is still missing. Checked: `datasets/raw/` (empty), a
  `find` over `~` excluding `Library/` and `.venv/` (no `.parquet` anywhere),
  and Spotlight (`mdfind`, none). Drive was not re-checked; the 2026-08-30
  audit found only `pilot_v0.npz` there.

## B. Findings fixed in code

### B.1 Falls labelled per segment, not per event (would have poisoned the safety-critical class)
Protocol: a 5-minute falling segment with a fall every ~15 s. Assembly
labelled every window in the segment `falling`. A fall lasts ~1.5 s, so
~90% of "falling" windows were standing, lying on the mattress, or getting
up. The model would have learned "lying on a mattress" as falling. The
realtime fast path (§2.3) would then have fired alerts on lying down.
Fall-detection datasets label short event-centred spans instead, e.g. 3 s
windows placed around a fall reference point (see the IntechOpen 2025
comparison and FallDeWideo). **Fix:** `session_script` cues each fall and
records cue times as `labels.json` `events`. A window is `falling` only if
it contains `[cue+0.5, cue+2.0]` s (`PreprocessConfig.fall_onset_s/
fall_end_s`, a calibration knob). Other falling-span windows are dropped.
`datasheet` counts falls, and `collector --label falling` is refused.
Caught before any fall was recorded, so nothing has to be re-collected.
**Calibrate on the first fall session:** plot motion energy around the cues
and move the two knobs so the fall peak sits inside the span.

### B.2 The default detrend deletes breathing — the cue for "motionless person vs. empty room"
`detrend_moving_mean(window=101)` is a 1 s moving-mean high-pass. Measured
gain of the actual filter at 100 Hz:

| f (Hz) | 0.10 | 0.20 | 0.25 | 0.33 | 0.50 | ≥1.0 |
|---|---|---|---|---|---|---|
| gain (dB) | −35.5 | −23.6 | −19.9 | −15.2 | −8.6 | 0.0 |

Adult resting breathing is 0.2–0.33 Hz, attenuated 15–24 dB. ESP32
presence systems detect a motionless person from the 0.08–0.6 Hz band over
~30 s windows, against an idle baseline (esphome-wifi-csi; Espressif
esp-csi presence examples). Our pipeline removes that band, then looks
through a 3 s window, shorter than one breath. So background, standing,
sitting and lying separate only by static per-subcarrier attenuation.
Pilot 1 showed that attenuation drifts between sessions: background fell
from 1.00 to 0.49 across sessions, and 85 empty-room windows were
predicted `standing`. Expect the same collapse for the static postures in
cross-session and cross-subject splits.
**Fix (enabling, not a claim):** `python -m csihar.dataset
--detrend-window N` makes the choice explicit and recorded. `N=3001` passes
the band flat (≤0.4 dB). Checkpoints now carry the preprocessing, and
serving rebuilds it, so a model trained on a non-default detrend cannot be
served with the default one.
**Experiment to run on pilot session 2** (a report ablation, "does
breathing carry presence?"):
```bash
python -m csihar.dataset --out ds_presence.npz --window-s 20 --hop-s 5 --detrend-window 3001
python -m csihar.baseline --data ds_presence.npz   # band_energy_0_2hz carries breathing
```
If background vs. static cross-session accuracy improves materially, add a
long-context presence stage in front of the 3 s activity model (two
timescales). If it does not, report static-posture/background confusion as
a physical limit of single-antenna amplitude sensing — a legitimate result,
not a failure.

### B.3 Other fixes
- `make_windows`: 50 s → 0.24 s per receiver for a 25-min session
  (re-sorted and copied the whole stream per window). Output is
  bit-identical.
- Training peak memory: 3.73 → 1.73 GB on a 0.56 GB dataset (per-batch
  normalization). The old path extrapolates to ~30 GB at the 25k-window
  target and would have OOMed Colab.
- `python -m csihar.dataset` assembly CLI. Before this, every `.npz` was
  built ad hoc. The npz now stores provenance: PreprocessConfig, git SHA,
  and session list.
- The dashboard ignored the checkpoint's window length: the §2.4 fix had
  covered only `realtime`. Both now use `serving_preprocess_config`.
- Report build: the placeholder figure macro produced 16 LaTeX errors
  (underscores in filenames). Fixed; pdflatex now builds with 0 errors.
- Deleted `preprocessing/normalize.py` (deferred item from §3 above).

## C. Direction check

**Keep:** esp-csi on S3, amplitude only, three-split evaluation, the
classical baseline gate, window-level alignment, the fall fast path. These
match the literature and the evidence so far. The host software is
complete, tested (252 tests), and now fast enough for the full dataset.
**Stop building software.** Every remaining deliverable is gated on data.

**Change:**
1. **Schedule is the top risk.** Data is at ~5% with Phase 3's window
   closing and Dashain/Tihar ahead. M3 at 25k windows ≈ 24 scripted
   subject-sessions of ~30 min (8 subjects × 3). That is roughly 20 hours
   of recording, feasible in 5–6 weeks at 4–5 sessions per week. Start
   with the 4 team members now. Schedule outside subjects for after Tihar.
   If slipping, cut to 6 subjects × 3 sessions before cutting sessions per
   subject: cross-session is the split the ≥85% target is judged on.
2. **Dedicated AP on channel 11 before session 2**, not later. Changing the
   radio mid-dataset adds a confound. Channel 6 has a −45 dBm neighbour.
3. **Expect static-posture classes to limit the headline number.** Plan the
   results chapter around it: report motion classes (walking, falling) and
   static classes separately, and run B.2's presence experiment.
4. **On-device int8 inference stays a stretch goal** until M4 is done. The
   feasibility study already shows it needs architecture changes.

## D. Sources (retrieved 2026-09-30)

- esphome-wifi-csi (breathing band 0.08–0.6 Hz, ~32 s DFT window, idle
  baseline): https://github.com/PeterkoCZ91/esphome-wifi-csi
- Espressif esp-csi presence examples: https://github.com/espressif/esp-csi
- Real-time fall detection via Wi-Fi CSI, CNN-LSTM/GNN/Transformer
  comparison (event-segment windows): https://www.intechopen.com/journals/1/articles/693
- FallDeWideo (video-aided fall labelling): https://dl.acm.org/doi/10.1145/3615984.3616501
- ESP32 LSTM HAR, >93% under 5-fold CV — the random-split regime:
  https://www.atlantis-press.com/proceedings/sasi-ite-25/126021002
- CSI-Chain (ESP32 end-to-end framework, 2026): https://itiis.org/digital-library/106117

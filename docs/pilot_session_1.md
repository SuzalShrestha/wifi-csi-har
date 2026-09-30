# Pilot session 1 — 2026-08-27/28, room_a, subject `sujal`

First real data. Purpose per IMPLEMENTATION_PLAN Phase 2 item 4: expose
pipeline bugs and class confusions *before* the full collection effort.
It did.

## What was recorded

| session | label | duration | notes |
|---|---|---|---|
| `20260827_232028_sujal_background` | background | 5 min | room empty |
| `20260827_232655_sujal_scripted` | walking, standing, sitting, lying | 4 × 5 min | one contiguous block each, in that order |
| `20260827_235706_sujal_background` | background | 5 min | room empty, boards untouched since 23:20 |

Rejected: `20260827_230632_sujal_scripted` (moved to `datasets/rejected/`) —
the subject walked during both the `background` and `walking` segments, so
both labels describe the same activity.

Excluded: `20260827_214816_bringup_background`. Its own notes said "room NOT
controlled, do not use for training", but that was free text no code read, so
`assemble_dataset` swept it in and it contributed ~18 background windows from
an uncontrolled room to the first pass of this analysis. It now carries
`exclude_from_dataset: true`, which `assemble_dataset` and `datasheet` both
honour. Re-running without it gave **identical** numbers (0.717 overall,
background 98/199), so nothing below depends on the contamination.

## Capture quality — all PASS

100.36–100.46 Hz on every receiver, **0 sequence gaps, 0 parse errors**
across 20 minutes and 123k frames per receiver. `null_mismatch=False`
throughout. Channel 6 / BW20 held for the entire session with `drift=False` —
the ISP router's auto-select did **not** hop over a 20-minute window, which
was the main open risk from the channel constraint.

## Results

Assembled dataset: 1207 windows, `(n, 3, 300, 52)`.
Logged in `experiments/results/baseline.csv` (git SHA `f5eb75a`, seed 0).

| split | model | accuracy | trustworthy? |
|---|---|---|---|
| random | svm_rbf | 0.992 | no — see below |
| random | random_forest | 0.979 | no |
| temporal (last 25% of each block) | random_forest | 0.940 | partly |
| cross-session | both | ~0.00 | **no — degenerate split** |

### The random split is not a result

Each activity was recorded as one contiguous 5-minute block, so "which
activity" is perfectly confounded with "what time it was". A model can score
94–99% by learning slow radio drift across the 20 minutes. Holding out the
last 25% of each block instead of interleaving drops it to 0.940, but that
is still within one session and does not break the confound.

### Background does not generalise across sessions — the real finding

Background was recorded twice, before and after the activities, boards
untouched. Training on the *before* session and testing on the *after* one:

| | accuracy |
|---|---|
| background, tested within its own session | 1.00 |
| background, tested on a different session | **0.49** |
| overall (5 classes) | 0.954 → **0.717** |

85 empty-room windows were classified as `standing`. The model had learned
"this is what the room sounded like at 23:57", not "nobody is here".

Motion energy explains why: empty room reads 1.14 / 2.67 / 0.83 (rx1/2/3)
and a motionless person reads 1.07 / 2.57 / 0.91. Nothing moves in either
case; the only difference is a static body attenuating certain subcarriers,
and that signature drifts between sessions.

This does **not** show the task is impossible — it shows one session of
background gives the model no way to separate the incidental from the
person. It is the direct justification for the plan's ≥3 sessions per
subject.

### Cross-session was degenerate, not catastrophic

`split_cross_session` held out both background sessions, leaving a test set
containing only `background` while training had all five classes. Accuracy
~0.00 is uninterpretable, not a failure. `dataset.split_coverage_note` now
detects this, warns, and writes the reason into the results CSV so the number
cannot be misread later.

## Colab GPU run — 2026-08-28

Both deep models were retrained on Colab (Tesla T4) from
`MyDrive/wifi-csi-har-data/pilot_v0.npz`. Raw output preserved verbatim in
`experiments/results/dl_colab.csv`; checkpoints and confusion matrices copied
to `experiments/checkpoints/` and `docs/figures/colab/`.

| model | split | accuracy | macro-F1 | trustworthy? |
|---|---|---|---|---|
| cnn | random | 0.9370 | 0.7717 | no — same temporal confound as above |
| cnn_lstm | random | 0.9244 | 0.7616 | no — same |
| cnn | cross-session | 0.9749 | 0.1645 | **no — degenerate split** |
| cnn_lstm | cross-session | 0.9598 | 0.1632 | **no — degenerate split** |

### The cross-session accuracy of 0.97 is an artifact, not a result

`split_cross_session` put session `20260827_235706_sujal_background` in test:
199 windows, **all of them `background`**, against 989 training windows
covering all five classes. Scoring 97.5% on a single-class test set measures
nothing, and the macro-F1 of 0.164 is the giveaway — four of six classes have
no support, so they contribute zero regardless of the model.

**The checkpoints themselves are sound.** Over all 1188 windows
`cnn_cross-session_0.pt` predicts a healthy spread
(background 393, walking 200, standing 200, sitting 198, lying 197) — it did
not collapse to the majority class. It trained on more data than the
random-split model (989 vs 950 windows) and is the better checkpoint to demo.
It is the *metric* that is void, not the weights.

The honest summary is that **this dataset cannot produce a cross-session
number at all.** Only one session contains activities; hold it out and the
test set is empty of them, keep it in and there is nothing left to hold out.
Session 2 has to contain all five activities for the split to mean anything.

### Two guardrail gaps this exposed

1. `split_coverage_note` was wired into `baseline.py` but **not** `train.py`,
   so the deep-learning path wrote a 0.9749 row with an empty `notes` column
   that reads exactly like a real result. Now fixed and mutation-tested.
2. The results `config` column recorded `"device": "auto"` rather than the
   device that ran. Since CPU and MPS give different accuracy on identical
   data and seeds, `"auto"` is exactly as useless as no field at all — the
   whole point of committing the device was cross-machine comparability. The
   resolved device is now written instead.

`git_sha` is `unknown` in `dl_colab.csv` because Colab runs from a Drive copy
with no `.git`. Clone the repo into Drive instead of copying it if that column
needs to be meaningful.

## Realtime engine vs offline pipeline — 2026-08-28

An earlier note in this session claimed the streaming engine scored 0.567 on
a session the CNN was trained on, against 0.903 offline. **That figure was
wrong** — it mislabelled the predictions. Measured properly, replaying
`20260827_232655_sujal_scripted` through `RealtimeEngine` and scoring with
the same boundary rule `assemble_session` uses:

| path | accuracy | windows |
|---|---|---|
| offline (`assemble_session`) | 0.9177 | 790 |
| realtime, raw label | 0.8671 | 790 scoreable of 817 emitted |
| realtime, smoothed label (before fix) | 0.7127 | " |
| realtime, smoothed label (after fix) | 0.8025 | " |

### The 5-point raw gap is causality, not a bug

Window-for-window the two paths agree on 96% of predictions. The residual
comes from `detrend_moving_mean` being a **centered** 101-tap kernel: offline,
every interior sample sees 0.5 s of future data; in the stream that future
has not happened. Mean absolute difference is 0.0001 mid-window and 0.91 in
the last 0.5 s. `_slice_window` already takes 0.5 s of *lead-in*, which is
why the leading edge is clean — only the trailing half is unfixable without
delaying every prediction by 0.5 s. Worth knowing, not worth paying for.

### The smoother was strictly worse than no smoothing

`smooth_predictions` mapped any window below `min_confidence` to the literal
label `"unknown"`, which then **competed in the majority vote**. Two unsure
windows could veto three confident ones. Measured over the pilot session:

| | accuracy | label changes / 20 min | says "unknown" |
|---|---|---|---|
| raw | 0.9177 | 29 | 0% |
| smoothed, old | 0.8557 | **29** | 10.1% |
| smoothed, abstaining | 0.9051 | **19** | 2.8% |

Identical flicker to raw and 6 points worse — it bought nothing. Low
confidence now abstains: the entry drops out of the vote, and `"unknown"` is
returned only when nothing recent is confident. Ground truth changes label 3
times in those 20 minutes, so 19 is still far from clean.

**Report the raw label's accuracy.** Even fixed, smoothing costs accuracy
(0.8025 vs 0.8671 end-to-end) because these errors cluster rather than being
independent noise. Its value is display stability. The dashboard streams both.

## What this changes for session 2

1. **Interleave background with the activities in the same session.** Use
   `--lead-in 15` so a lone subject can press Enter and leave the room before
   the labelled span starts.
2. **Reverse the activity order** (`lying, sitting, standing, walking`). If
   time-order drift were doing the work, it cannot survive being flipped.
3. Record on a **different day**, with the board layout unchanged.
4. Until then, no accuracy number from this data belongs in the report.

## Collection progress

`python -m csihar.datasheet datasets/raw`:

```
class            seconds   minutes  ~windows
background           601      10.0       399
standing             300       5.0       199
sitting              300       5.0       199
lying                300       5.0       199
walking              300       5.0       199
falling                0       0.0         0   <- MISSING
TOTAL               1801      30.0      1195

  [--] subjects                    1 / 8
  [--] windows                  1195 / 25000
  [--] environments                1 / 2
  [--] sujal distinct days         1 / 3
```

About 5% of the M3 window target, from one subject on one day.

## Still outstanding

- Device layout (positions/heights) is still unrecorded in
  `collection_protocol.md`. Session 2 is only comparable if the boards are
  where they were.
- No `falling` data yet — needs a mattress and a spotter.
- Second subject not yet recorded.

---

## Every pilot macro-F1 is measured against a 0.8333 ceiling — 2026-08-28

Found while running the first ablation smoke test. `compute_metrics`
averages F1 over the full label vocabulary (6 classes), and `falling` has
zero windows. An absent class scores F1 = 0 with `zero_division=0`, and no
model can avoid it.

**A perfect classifier on this data scores macro-F1 0.8333.** Verified
directly:

```
compute_metrics(y, y, names)  # identical predictions, 'falling' absent
  accuracy  1.0
  macro_f1  0.8333
```

Which changes how the existing pilot table reads:

| model | split | accuracy | macro-F1 | % of ceiling |
|---|---|---|---|---|
| svm_rbf | random | 0.9917 | 0.8261 | **99.1%** |
| random_forest | random | 0.9793 | 0.8143 | 97.7% |
| cnn (Colab T4) | random | 0.9370 | 0.7717 | 92.6% |
| cnn_lstm (Colab T4) | random | 0.9244 | 0.7616 | 91.4% |

Two consequences.

**1. No pilot macro-F1 is comparable to any later run.** Once `falling` is
collected the ceiling moves to 1.0. A future 0.85 would look like an
improvement over the baseline's 0.8261 while actually being worse relative
to what is achievable.

**2. The classical baselines beating the deep models is not (yet) evidence
of a broken pipeline.** The project rule says a DL model that cannot beat
SVM/RF signals a defect. That rule's diagnostic power applies on a
non-leaky split with enough data. Here the split is maximally leaky
(50%-overlapping windows from one subject on one day), and memorising a
leaky split is precisely what an RBF-kernel SVM does best. Re-check the rule
once cross-session is interpretable; do not act on it now.

`dataset.label_coverage_note` now detects this, `train` and `baseline` warn
and write it into the results CSV's `notes` column. Rows predating
2026-08-28 do not carry the note — apply the ceiling by hand when reading
them.

Note that `split_coverage_note` never caught this and could not have: it
compares the train class set against the test class set, and an unpopulated
class is missing from both.

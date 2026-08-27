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

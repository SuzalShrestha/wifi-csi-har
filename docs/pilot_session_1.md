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

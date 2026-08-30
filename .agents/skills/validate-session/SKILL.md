---
name: validate-session
description: Run QA on a recorded CSI session before accepting it into the dataset.
---

# Validate Session

Run this **immediately after every `/collect-session`**, before the session
is labeled "good" or shared. Never accept a session into the dataset without
a passing QA report.

## Command

```bash
.venv/bin/python -m csihar.qa datasets/raw/<session_dir>
```

Optional thresholds (defaults shown):

```bash
.venv/bin/python -m csihar.qa datasets/raw/<session_dir> \
    --min-rate-hz 90.0 --max-gap-fraction 0.05 --min-mcs-fraction 0.9
```

This prints one line per receiver (`rx1`, `rx2`, `rx3`, ...) and saves an
amplitude heatmap PNG (`<receiver>_qa_heatmap.png`) into the session
directory for each one. Exit code is `1` if any receiver failed, `0` if all
passed.

## Reading a report line

```
rx1: n_frames=286 duration_s=2.99 rate_hz=95.31 gap_frac=0.0467 rssi_mean=-42.00 rssi_std=0.00 null_mismatch=False -> PASS
  radio: ch=6 bw=0 drift=False mcs=7 mcs_frac=1.000
```

- **n_frames / duration_s**: sanity-check these match what you expect from
  the recording length.
- **mean_rate_hz**: average packet rate. Firmware targets 100 Hz.
- **seq_gap_fraction**: fraction of missing sequence numbers between the
  first and last frame — captures serial drops the simulator/collector
  didn't already discard.
- **rssi_mean / rssi_std**: rough link-quality sanity check, not gated on by
  default.
- **radio: ch / bw / drift**: the channel and bandwidth the session ran on,
  and whether either one *changed* mid-session. Both must be pinned across
  the whole dataset, so `drift=True` means the frames straddle two radio
  environments.
- **radio: mcs / mcs_frac**: the dominant modulation-and-coding scheme and
  the fraction of frames at it. Unlike channel, MCS is chosen per packet by
  the router's rate adaptation, so it is the *fraction* that matters.
- **`-1` / `nan` in the radio line**: that column is absent, i.e. the session
  predates `mcs`/`bandwidth` being persisted. Those checks are skipped rather
  than failed.
- **null_subcarrier_mismatch**: `True` means `detect_null_subcarriers` found
  empirically-dead subcarriers *inside* the theoretical usable band
  (`usable_lltf_indices`) — i.e. the buffer-ordering assumption may be wrong
  for this hardware.

## Thresholds and what failures usually mean

- **Low `mean_rate_hz` (< 90 Hz)**: baud mismatch (must be 921600, not
  115200) or a flaky/low-bandwidth USB hub. Re-check `idf.py menuconfig`
  console baud and try a different USB port/cable.
- **High `seq_gap_fraction` (> 0.05)**: serial drops — same causes as above,
  or the host was under load and fell behind reading the port.
- **`drift = True`**: the router changed channel or bandwidth during the
  recording. On an auto-channel ISP router (see
  `docs/collection_protocol.md`) this is the expected failure mode. The
  session is **not** salvageable — discard and re-record, and if it recurs
  switch to a dedicated AP where the channel can be pinned.
- **Low `mcs_frac` (< 0.9)**: the link was unstable enough that rate
  adaptation kept switching, which changes the CSI's effective SNR partway
  through. Usually means interference or a marginal RSSI — check receiver
  placement and the channel survey before re-recording.
- **`null_subcarrier_mismatch = True`**: the subcarrier ordering assumption
  in `preprocessing/subcarriers.py` doesn't match this receiver's real
  output. **Escalate**: run `detect_null_subcarriers` directly against this
  receiver's amplitude data, compare the indices against
  `usable_lltf_indices()`, and resolve the disagreement (see
  `csihar/preprocessing/subcarriers.py` docstring) before collecting more
  data on this board — don't just re-record and hope.
- **`<2 frames` / immediate fail with `n_frames` reported**: the receiver
  effectively didn't capture anything; check the board's serial connection
  before re-recording.

## Rule

**A failed session is discarded and re-recorded the same day** — don't keep
partial or borderline sessions "just in case." Also **eyeball every saved
heatmap PNG**: even a numerically passing session can lack visible activity
structure (e.g. subject not actually moving, board pointed wrong way).
Passing stats + a heatmap with no visible pattern is still a reason to
re-record.

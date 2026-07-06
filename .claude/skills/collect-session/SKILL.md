---
name: collect-session
description: Use when recording a labeled CSI data-collection session with the three ESP32-S3 receivers.
---

# Collect Session

Procedure for running one labeled data-collection session with rx1/rx2/rx3.

## 1. Pre-flight

- All three boards are flashed, joined to the router, and streaming
  `CSI_DATA` lines (see `flash-firmware` skill if not).
- Tripod/device positions match the layout recorded in
  `docs/collection_protocol.md` (rx1/rx2/rx3 + router position and height).
  Re-measure and re-tape tripod feet if this is a new room or the layout
  drifted.
- Photograph the device layout before the session.
- Note/confirm the router's WiFi channel (fixed 2.4 GHz, HT20, band
  steering off) in `docs/collection_protocol.md`.

## 2. Run the collector

```bash
.venv/bin/python -m csihar.collector \
  --port /dev/cu.usbmodemXX1=rx1 --port /dev/cu.usbmodemXX2=rx2 \
  --port /dev/cu.usbmodemXX3=rx3 \
  --label walking --subject sujal --env room_a --duration 300
```

- `--port DEV=ID` is repeatable, one per receiver (`DEV` = serial port,
  `ID` = receiver label rx1/rx2/rx3).
- Valid `--label` values: `walking`, `sitting`, `standing`, `lying`,
  `falling`, `background`.
- `--subject` is the subject's name/id, `--env` the room/environment tag,
  `--duration` in seconds, `--out` defaults to `datasets/raw`.

Per-session script (operator reads aloud, subject follows):
1. Start the collector with the correct `--label --subject --env`.
2. 10 s of `background` with the subject outside the room — sanity
   reference.
3. 5 min of the target activity.

### Fall sessions — safety rules

- Falls happen onto a mattress.
- A spotter is present at all times.
- Alternate fall direction each rep: forward, backward, sideways.
- Space repetitions ~15 s apart.

## 3. During the session

The collector prints a live status line every ~2 s:

```
[ 12.3s] frames rx1:1203 rx2:1199 rx3:1187
```

Frame counts across receivers should climb together at roughly the same
rate. A receiver stalling relative to the others signals a dropped
connection — stop and investigate rather than let the session finish.

## 4. End-of-session summary and acceptance

At the end, the collector prints per-receiver:

```
rx1: 30000 frames @ 99.8 Hz, 3 seq gaps, 1 parse errors -> <path>
```

Accept the session only if, for every receiver:
- Rate is close to 100 Hz (the firmware's `CONFIG_SEND_FREQUENCY`).
- Seq gaps and parse errors are low single digits percent of total frames.

A `WARNING: <receiver> captured 0 frames` means that receiver failed
entirely — discard the session.

## 5. Immediate post-check

Replay one receiver's parquet right away:

```bash
.venv/bin/python -m csihar.view --replay datasets/raw/<session>/rx1.parquet
```

If the heatmap looks wrong, or any receiver logged high loss, discard the
session.

## 6. Bad session rule

A session that looks wrong (bad heatmap, high seq gaps/parse errors, a
0-frame receiver) is discarded and redone the same day — do not keep
partial or suspect sessions in `datasets/raw`.

## 7. Before accepting the data

Run `/validate-session` on the new session directory before accepting the
data.

# Data Collection Protocol (v0 — finalize in Phase 2)

## Fixed parameters
- Packet rate: 100 Hz per receiver, driven by host UDP downlink traffic
  (`--traffic <ip>` per receiver — without it CSI drops to <1 Hz; see
  CLAUDE.md hard-won facts). Firmware ping (`CONFIG_SEND_FREQUENCY`) alone
  is not sufficient with a real router.
- Serial: 921600 baud, host timestamps at arrival
- Window: 3 s, 50% overlap (set in preprocessing, not at capture time)
- Router: channel __, HT20, model __________ (fill in after channel survey)

## Device layout (photograph + measure before EVERY session)
- rx1: position ____ height ____
- rx2: position ____ height ____
- rx3: position ____ height ____
- Router: position ____ height ____
- Mark tripod feet positions on the floor with tape.

## Classes
walking · sitting · standing · lying · falling · background (empty room)

Transitions (sit-down/stand-up) are NOT separate classes in v1; trim them
from window labels during preprocessing.

## Per-session script (operator reads aloud, subject follows)
1. Start collector with correct `--label --subject --env`
2. 10 s background (subject outside room) — sanity reference
3. 5 min of the target activity (falls: onto mattress, spotter present,
   alternate forward/backward/sideways, ~15 s spacing)
4. Run `python -m csihar.view --replay <session>/rx1.parquet` immediately;
   discard and redo the session if the heatmap looks wrong or any receiver
   logged high loss

## Targets
- 8–10 subjects, ≥3 sessions each on different days
- ≥5 min per activity per subject
- One extra half-session in a second room (transfer test set)

## Session command template
```bash
.venv/bin/python -m csihar.collector \
  --port /dev/cu.usbmodemXX1=rx1 --port /dev/cu.usbmodemXX2=rx2 \
  --port /dev/cu.usbmodemXX3=rx3 \
  --traffic <rx1-ip> --traffic <rx2-ip> --traffic <rx3-ip> \
  --label walking --subject sujal --env room_a --duration 300
```

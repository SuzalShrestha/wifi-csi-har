# Data Collection Protocol (v0 — finalize in Phase 2)

## Fixed parameters
- Packet rate: 100 Hz per receiver, driven by host UDP downlink traffic
  (`--traffic <ip>` per receiver — without it CSI drops to <1 Hz; see
  CLAUDE.md hard-won facts). Firmware ping (`CONFIG_SEND_FREQUENCY`) alone
  is not sufficient with a real router.
- Serial: 921600 baud, host timestamps at arrival
- Window: 3 s, 50% overlap (set in preprocessing, not at capture time)
- Router: SSID `shrestha`, BSSID `60:bd:2c:2e:f8:19` (2.4 GHz radio),
  **channel 6, BW20**, WPA2-PSK, model __________
  (measured 2026-08-27 from the receivers' own association logs).
  **The channel cannot be pinned on this router** — it is an ISP-supplied
  (WorldLink) unit on auto-select, so channel 6 is wherever auto-select
  landed and a mid-session hop is possible. Mitigation, in preference order:
  (a) check the router admin UI anyway — many WorldLink ONTs (ZTE F660/F670L,
  Huawei HG8145) do expose channel selection under WLAN → Basic even on the
  subscriber account; (b) use any spare router as a dedicated AP — it needs
  no internet connection, only to transmit, and this also solves band
  steering and other clients in one move; (c) accept auto-select and rely on
  the drift detection below. Still disable band steering and move other
  clients off this SSID if the UI allows it.

  Every session is now checked for channel/bandwidth drift by
  `csihar.qa` (`radio: ... drift=`) — a session whose frames straddle two
  channels fails QA and is discarded rather than silently entering the
  dataset. `channel` held constant across all three receivers for the full
  30 s bring-up capture, so short-timescale hopping is not happening; a
  20-minute collection session is the untested case.

### Channel survey — 2026-08-27
41 scans over ~7 min from the host's `en0` at the collection site.
Own AP excluded from the interference figures — it is signal, not noise.

| ch | strongest *other* AP | verdict                                   |
|----|----------------------|-------------------------------------------|
| 1  | -82 dBm              | usable fallback                           |
| 5  | -73 to -78 dBm       | overlaps 6                                |
| 6  | **-45 dBm**          | **current — a neighbour only 4 dB below our own AP** |
| 8  | -93 dBm              | overlaps 6/11                             |
| 11 | -89 dBm (often undetected) | **recommended**                     |

**Channel 11 is the target if the channel ever becomes settable** (see the
router note above — it currently is not). Channel 6 shares the band with a
neighbour at -45 dBm against our own -41 dBm, i.e. near-equal power
co-channel contention — that AP's traffic both steals airtime and adds
uncontrolled multipath variation to every window we record. Channel 11 would
improve the AP-interference floor by roughly 44 dB. This is the single
strongest argument for option (b) above, a dedicated AP.

Caveats on that number: the cleanest reading for ch 11 was "no AP detected",
so 44 dB is computed against the weak -89 dBm AP seen in the longer scan,
not against a true noise floor. A passive AP scan also cannot see
non-802.11 emitters (microwave ovens, ZigBee, analogue video). If periodic
~2.45 GHz disruption shows up in captures, fall back to **channel 1**, which
sits furthest from typical microwave-oven emission. Finally this survey was
taken from the laptop's position; the receivers sit elsewhere in the room,
so re-check if their RSSI to the router changes markedly after the move.

The receivers follow the SSID, so a channel change — or a swap to a
dedicated AP broadcasting the same SSID and passphrase — needs **no
reflash**: just power-cycle them and re-confirm IPs (DHCP leases may shift).

## Receiver identity (stable — verified 2026-08-27)
Serial paths come from the CH343 UART bridge serial number, so they persist
across reboots and re-plugs. **Use the UART USB-C port, not the native USB
port** — the firmware's primary console is UART0 @ 921600, which is the path
verified at 100 Hz / 0% loss. The native USB port also emits CSI over a
115200 secondary console, so it *looks* like it works; it was only ever
observed unloaded (0.2 Hz) and has not been validated at 100 Hz. Don't
collect on it.

| id  | serial port                  | IP            | STA MAC             | RSSI  |
|-----|------------------------------|---------------|---------------------|-------|
| rx1 | `/dev/cu.usbmodem5B5E0807741`| 192.168.1.97  | `1c:db:d4:43:f1:d0` | -55   |
| rx2 | `/dev/cu.usbmodem5C842982391`| 192.168.1.98  | `a4:cb:8f:f8:39:a8` | -61   |
| rx3 | `/dev/cu.usbmodem5C842990031`| 192.168.1.99  | `a4:cb:8f:f8:34:bc` | -66   |

IPs are DHCP leases — re-check them (or set static reservations in the
router) before each session; the collector needs them for `--traffic`.
Label the three boards physically with rx1/rx2/rx3 to match this table.

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
  --port /dev/cu.usbmodem5B5E0807741=rx1 \
  --port /dev/cu.usbmodem5C842982391=rx2 \
  --port /dev/cu.usbmodem5C842990031=rx3 \
  --traffic 192.168.1.97 --traffic 192.168.1.98 --traffic 192.168.1.99 \
  --label walking --subject sujal --env room_a --duration 300
```

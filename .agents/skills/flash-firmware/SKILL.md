---
name: flash-firmware
description: Use when setting up ESP-IDF, flashing an ESP32-S3 receiver with the esp-csi csi_recv_router firmware, or debugging a board that isn't streaming CSI_DATA lines.
---

# Flash Firmware

Procedure for building and flashing the `csi_recv_router` firmware onto an
ESP32-S3 CSI receiver board, and verifying it streams data correctly.

## 1. Check prerequisites

```bash
command -v cmake && command -v ninja
```

If missing:

```bash
brew install cmake ninja dfu-util ccache
```

## 2. Install ESP-IDF (one-time, idempotent)

```bash
./scripts/setup_espidf.sh
```

This clones ESP-IDF v5.3.2 into `~/esp/esp-idf` (~2 GB) and runs
`install.sh esp32s3`. Safe to re-run.

Every new shell, before using `idf.py`:

```bash
source ~/esp/esp-idf/export.sh
```

## 3. Get esp-csi

`firmware/esp-csi/` is git-ignored. If it's missing, re-clone it:

```bash
git clone --depth 1 https://github.com/espressif/esp-csi firmware/esp-csi
git -C firmware/esp-csi apply ../patches/csihar.patch
```

The patch adds `esp_wifi_set_ps(WIFI_PS_NONE)` (default power save starves
CSI) and fixes `sdkconfig.defaults` to use modern IDF 5.x console symbols
(custom console UART0 @ 921600 — the example's own defaults use deprecated
names that are silently ignored, leaving the console on USB-JTAG at 115200).
Do not touch the `CSI_DATA` output code — the host parser
(`csihar/parser.py`) matches its CSV line format exactly.

## 4. Configure and flash (repeat per board)

```bash
source ~/esp/esp-idf/export.sh
cd firmware/esp-csi/examples/get-started/csi_recv_router
idf.py set-target esp32s3
idf.py menuconfig
```

In menuconfig, set:
- **Example Connection Configuration -> WiFi SSID / Password**: the router's
  credentials.

Console UART + 921600 baud come from the patched `sdkconfig.defaults`. If
editing an existing `sdkconfig` by hand instead: baud only sticks in
**custom console UART mode** (`CONFIG_ESP_CONSOLE_UART_CUSTOM=y` +
`CONFIG_ESP_CONSOLE_UART_CUSTOM_NUM_0=y` + baudrate) — in default mode
kconfgen silently resets `CONFIG_ESP_CONSOLE_UART_BAUDRATE` to 115200, and
the deprecated alias `CONFIG_CONSOLE_UART_BAUDRATE` later in the file
overrides hand edits (last value wins).

Find the port, then flash + monitor:

```bash
ls /dev/cu.usbmodem*
idf.py -p /dev/cu.usbmodem<N> flash monitor
```

## 5. Per-board checklist

- [ ] `CONFIG_SEND_FREQUENCY` is 100 (default in `app_main.c`)
- [ ] Console baud is 921600
- [ ] Board joins the router and `CSI_DATA,...` lines stream in the monitor
- [ ] Physically label the board rx1 / rx2 / rx3 with a marker, and note the
      port -> receiver ID mapping

## 6. Verify from the host

CSI only streams at ~100 Hz while the host floods the board's IP with UDP
(the router sends sparse frames at DSSS rates that produce no CSI — see
AGENTS.md). Get the board's IP from the boot log (`got ip:...`), then:

```bash
.venv/bin/python -m csihar.view --live /dev/cu.usbmodem<N> --traffic <board-ip>
```

Wave a hand between the board and the router — the heatmap must visibly
react. If it doesn't react, treat the board as not verified.

## Troubleshooting

- **No serial port shows up in `ls /dev/cu.usbmodem*`**: check the USB
  cable (must be data-capable, not charge-only) and that the board's USB
  driver is installed; try a different cable/port.
- **Garbage / unreadable lines in the monitor**: baud mismatch — confirm
  the monitor and `csihar.view --live` are both using 921600, and that
  console baud was actually changed in menuconfig (not left at default).
- **Board joins WiFi but no `CSI_DATA` lines appear**: wrong SSID/password
  in menuconfig, or the router is 5 GHz-only / band-steering the board onto
  5 GHz — the ESP32-S3 needs the 2.4 GHz band. Confirm the router has a
  fixed 2.4 GHz channel with band steering / smart connect OFF.
- **Frequent parse errors at 100 Hz**: baud too low (still 115200) or a
  marginal USB cable; re-check menuconfig and re-flash.
- **CSI_DATA streams but at <1 Hz**: no UDP downlink traffic — the
  firmware's own ping is not enough with a real router (replies go out at
  DSSS/CCK rates, which carry no OFDM LTF). Run `csihar.traffic` /
  `--traffic <board-ip>` on the host. Verified: 100 pkt/s of 200-byte UDP
  → 100 Hz CSI. Not an IDF version issue.

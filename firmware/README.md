# Firmware — ESP32-S3 CSI receiver

We use the official [espressif/esp-csi](https://github.com/espressif/esp-csi)
`csi_recv_router` example unmodified (cloned into `firmware/esp-csi/`,
git-ignored). Each of the three S3 boards pings the router at 100 Hz and
prints one `CSI_DATA` CSV line per reply over USB-serial. The host parser
(`csihar/parser.py`) matches this format exactly.

## One-time setup

1. Install ESP-IDF v5.x: run `../scripts/setup_espidf.sh` (installs to
   `~/esp/esp-idf`, ~2 GB download) then `source ~/esp/esp-idf/export.sh`.
2. Re-clone esp-csi if missing:
   `git clone --depth 1 https://github.com/espressif/esp-csi firmware/esp-csi`

## Configure and flash (per board)

```bash
source ~/esp/esp-idf/export.sh
cd firmware/esp-csi/examples/get-started/csi_recv_router
idf.py set-target esp32s3
idf.py menuconfig
#   Example Connection Configuration -> WiFi SSID / password  (the router)
#   Component config -> ESP System Settings -> Channel for console output
#     -> keep default USB Serial/JTAG or UART0 as wired on your devkit
idf.py -p /dev/cu.usbmodem<N> flash monitor
```

Checklist per board:
- [ ] `CONFIG_SEND_FREQUENCY` is 100 (default in app_main.c)
- [ ] Console baud raised to **921600** (menuconfig → Serial flasher config /
      console baud) — 115200 drops lines at 100 pkt/s
- [ ] Board joins the router and `CSI_DATA,...` lines stream in the monitor
- [ ] Note the port ↔ receiver ID mapping (rx1/rx2/rx3) — label the boards
      physically with a marker

## Router configuration (do once, document in docs/)

- Fixed 2.4 GHz channel (survey first, pick the least congested), 20 MHz
  bandwidth ("HT20"), band steering / smart connect OFF
- Dedicated SSID with no other clients if possible
- Record model + firmware version + settings in `docs/collection_protocol.md`

## Verify from the host

```bash
.venv/bin/python -m csihar.view --live /dev/cu.usbmodem<N>
```

Wave your hand between the board and the router: the heatmap must visibly
react (milestone M1).

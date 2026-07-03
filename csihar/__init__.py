"""WiFi CSI human activity recognition on ESP32-S3.

Host-side pipeline: serial acquisition -> parsing -> preprocessing ->
windowing -> models. Firmware side lives in firmware/ (espressif/esp-csi,
csi_recv_router example).
"""

__version__ = "0.1.0"

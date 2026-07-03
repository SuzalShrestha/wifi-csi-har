"""Parse CSI_DATA lines emitted by esp-csi's csi_recv_router on ESP32-S3.

Firmware format (examples/get-started/csi_recv_router/main/app_main.c,
non-C5/C6 build, which is what the ESP32-S3 uses):

    type,id,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,
    aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,
    secondary_channel,local_timestamp,ant,sig_len,rx_format,len,first_word,data

The `data` field is a quoted JSON-like int array: "[b0,b1,...]" of `len`
signed bytes. Per the ESP-IDF WiFi CSI documentation each subcarrier is two
bytes, IMAGINARY part first then REAL part. For a 20 MHz HT frame the first
128 bytes (64 complex values) are the LLTF block.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_METADATA_FIELDS = 24  # fields before `data`, including the "CSI_DATA" tag
LLTF_N_SUBCARRIERS = 64

# Column names matching the firmware header line (minus the leading `type`).
METADATA_COLUMNS = (
    "seq", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", "smoothing",
    "not_sounding", "aggregation", "stbc", "fec_coding", "sgi", "noise_floor",
    "ampdu_cnt", "channel", "secondary_channel", "local_timestamp", "ant",
    "sig_len", "rx_format", "len", "first_word",
)

_INT_FIELDS = {name for name in METADATA_COLUMNS if name != "mac"}


@dataclass(frozen=True)
class CsiFrame:
    """One CSI measurement. Immutable; csi array is the raw complex values."""

    host_ts: float          # host arrival time, time.time() seconds
    seq: int
    mac: str
    rssi: int
    noise_floor: int
    mcs: int
    bandwidth: int
    channel: int
    local_timestamp: int    # device-local microseconds
    sig_len: int
    csi: np.ndarray         # complex64, all subcarriers in buffer order

    @property
    def lltf(self) -> np.ndarray:
        """First 64 complex values (LLTF block for 20 MHz)."""
        return self.csi[:LLTF_N_SUBCARRIERS]

    @property
    def amplitude(self) -> np.ndarray:
        return np.abs(self.lltf).astype(np.float32)


class CsiParseError(ValueError):
    """Raised for a malformed CSI_DATA line."""


def parse_line(line: str, host_ts: float) -> CsiFrame | None:
    """Parse one serial line. Returns None for non-CSI lines (logs, headers).

    Raises CsiParseError for lines that claim to be CSI_DATA but are corrupt
    (truncated serial output is common at high packet rates — callers should
    count these, not crash).
    """
    line = line.strip()
    if not line.startswith("CSI_DATA,"):
        return None

    head, sep, data = line.partition(',"[')
    if not sep or not data.endswith(']"'):
        raise CsiParseError("unterminated data array")

    fields = head.split(",")
    if len(fields) != N_METADATA_FIELDS:
        raise CsiParseError(
            f"expected {N_METADATA_FIELDS} metadata fields, got {len(fields)}"
        )

    meta = _parse_metadata(fields[1:])
    csi = _parse_csi_array(data[:-2], expected_len=meta["len"])

    return CsiFrame(
        host_ts=host_ts,
        seq=meta["seq"],
        mac=meta["mac"],
        rssi=meta["rssi"],
        noise_floor=meta["noise_floor"],
        mcs=meta["mcs"],
        bandwidth=meta["bandwidth"],
        channel=meta["channel"],
        local_timestamp=meta["local_timestamp"],
        sig_len=meta["sig_len"],
        csi=csi,
    )


def _parse_metadata(fields: list[str]) -> dict:
    meta: dict = {}
    for name, raw in zip(METADATA_COLUMNS, fields):
        if name in _INT_FIELDS:
            try:
                meta[name] = int(raw)
            except ValueError as exc:
                raise CsiParseError(f"non-integer field {name}={raw!r}") from exc
        else:
            meta[name] = raw
    return meta


def _parse_csi_array(body: str, expected_len: int) -> np.ndarray:
    try:
        raw = np.array(body.split(","), dtype=np.int16)
    except ValueError as exc:
        raise CsiParseError("unparseable data array") from exc
    if raw.size != expected_len:
        raise CsiParseError(f"len field says {expected_len}, array has {raw.size}")
    if raw.size < 2 or raw.size % 2:
        raise CsiParseError(f"odd/empty CSI byte count {raw.size}")
    # Bytes come as (imaginary, real) pairs per the ESP-IDF CSI docs.
    imag = raw[0::2].astype(np.float32)
    real = raw[1::2].astype(np.float32)
    return (real + 1j * imag).astype(np.complex64)

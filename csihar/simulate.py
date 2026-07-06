"""Synthetic CSI generator emitting the exact csi_recv_router serial format.

Lets the entire host pipeline (parser -> collector -> preprocessing -> viz)
be developed and tested before hardware is on the desk, and provides
controlled fixtures for unit tests. The signal model is deliberately simple:
static per-subcarrier channel profile + activity-dependent sinusoidal
modulation + Gaussian noise + occasional impulsive spikes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .parser import LLTF_N_SUBCARRIERS, parse_line
from .storage import frames_to_dataframe, write_session_metadata

# Rough motion-frequency signatures (Hz) and modulation depths per activity.
ACTIVITY_PROFILES: dict[str, tuple[float, float]] = {
    "background": (0.0, 0.00),
    "standing":   (0.3, 0.02),
    "sitting":    (0.5, 0.03),
    "lying":      (0.4, 0.03),
    "walking":    (1.8, 0.25),
    "falling":    (4.0, 0.45),
}


def generate_lines(
    activity: str = "walking",
    duration_s: float = 10.0,
    fs: float = 100.0,
    mac: str = "aa:bb:cc:dd:ee:ff",
    seed: int = 0,
    spike_prob: float = 0.01,
    drop_prob: float = 0.02,
) -> list[tuple[float, str]]:
    """Return (timestamp, serial_line) tuples mimicking one receiver.

    Timestamps include jitter; a fraction of packets is dropped to mimic
    serial loss. Header/log lines are interleaved like real firmware output.
    """
    if activity not in ACTIVITY_PROFILES:
        raise ValueError(f"unknown activity {activity!r}")
    rng = np.random.default_rng(seed)
    freq, depth = ACTIVITY_PROFILES[activity]

    profile = _channel_profile(rng)
    lines: list[tuple[float, str]] = [(0.0, "I (1234) wifi: CSI RECV started")]
    n_packets = int(duration_s * fs)
    for i in range(n_packets):
        if rng.random() < drop_prob:
            continue
        t = i / fs + rng.normal(0, 0.001)
        modulation = 1.0 + depth * np.sin(2 * np.pi * freq * t + profile["phase"])
        amp = profile["amplitude"] * modulation + rng.normal(0, 0.5, LLTF_N_SUBCARRIERS)
        if rng.random() < spike_prob:
            amp = amp + rng.normal(0, 15.0, LLTF_N_SUBCARRIERS)
        lines.append((t, _format_line(i, mac, amp, rng)))
    return lines


def write_session(
    out_dir: Path,
    *,
    label: str,
    subject: str,
    environment: str = "sim_room",
    receivers: tuple[str, ...] = ("rx1", "rx2", "rx3"),
    duration_s: float = 10.0,
    seed: int = 0,
    t0: float = 1000.0,
) -> Path:
    """Write a complete simulated session dir in the on-disk storage layout.

    Creates ``out_dir/sim_<subject>_<label>_<seed>/`` containing one Parquet
    file per receiver plus ``metadata.json`` — the shared fixture factory for
    everything downstream of storage. Each receiver gets a distinct seed so
    the three streams are decorrelated; host_ts is ``t0 + line timestamp``
    (set at parse time — CsiFrame is frozen).
    """
    session_dir = out_dir / f"sim_{subject}_{label}_{seed}"
    write_session_metadata(
        session_dir,
        label=label,
        subject=subject,
        environment=environment,
        receivers={rx: "simulated" for rx in receivers},
        notes=f"simulate.write_session seed={seed} duration_s={duration_s}",
    )
    for i, rx in enumerate(receivers):
        lines = generate_lines(
            activity=label, duration_s=duration_s, seed=seed + 1000 * i
        )
        frames = []
        for ts, line in lines:
            frame = parse_line(line, host_ts=t0 + ts)
            if frame is not None:
                frames.append(frame)
        frames_to_dataframe(frames).to_parquet(session_dir / f"{rx}.parquet")
    return session_dir


def _channel_profile(rng: np.random.Generator) -> dict:
    idx = np.arange(LLTF_N_SUBCARRIERS)
    amplitude = 20 + 10 * np.sin(2 * np.pi * idx / 64 * 2 + rng.uniform(0, np.pi))
    amplitude[0] = 0.0                       # DC null
    amplitude[27:38] = 0.0                   # guard band nulls
    return {
        "amplitude": amplitude,
        "phase": rng.uniform(0, 2 * np.pi, LLTF_N_SUBCARRIERS),
    }


def _format_line(seq: int, mac: str, amplitudes: np.ndarray, rng) -> str:
    phases = rng.uniform(0, 2 * np.pi, LLTF_N_SUBCARRIERS)
    real = np.clip(amplitudes * np.cos(phases), -127, 127).astype(int)
    imag = np.clip(amplitudes * np.sin(phases), -127, 127).astype(int)
    interleaved = np.empty(2 * LLTF_N_SUBCARRIERS, dtype=int)
    interleaved[0::2] = imag  # firmware order: imaginary first
    interleaved[1::2] = real
    data = ",".join(str(v) for v in interleaved)
    meta = (
        f"CSI_DATA,{seq},{mac},-42,11,1,7,0,1,0,0,0,1,0,-92,0,6,0,"
        f"{seq * 10000},0,128,1,{2 * LLTF_N_SUBCARRIERS},0"
    )
    return f'{meta},"[{data}]"'

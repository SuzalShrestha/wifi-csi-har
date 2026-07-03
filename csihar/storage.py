"""Session storage: Parquet frames + JSON metadata sidecar.

Layout:  datasets/raw/<session_name>/
             metadata.json
             <receiver_id>.parquet     (one file per ESP32)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .parser import CsiFrame


def frames_to_dataframe(frames: list[CsiFrame]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "host_ts": [f.host_ts for f in frames],
            "seq": [f.seq for f in frames],
            "rssi": [f.rssi for f in frames],
            "noise_floor": [f.noise_floor for f in frames],
            "channel": [f.channel for f in frames],
            "local_timestamp": [f.local_timestamp for f in frames],
            "csi_real": [f.csi.real.astype(np.float32) for f in frames],
            "csi_imag": [f.csi.imag.astype(np.float32) for f in frames],
        }
    )


def load_amplitudes(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one receiver's parquet -> (host_ts (n,), amplitude (n, n_sc))."""
    df = pd.read_parquet(path)
    real = np.stack(df["csi_real"].to_numpy())
    imag = np.stack(df["csi_imag"].to_numpy())
    return df["host_ts"].to_numpy(), np.abs(real + 1j * imag)


def write_session_metadata(
    session_dir: Path,
    *,
    label: str,
    subject: str,
    environment: str,
    receivers: dict[str, str],
    notes: str = "",
) -> Path:
    session_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": label,
        "subject": subject,
        "environment": environment,
        "receivers": receivers,  # receiver_id -> serial port / position note
        "notes": notes,
    }
    path = session_dir / "metadata.json"
    path.write_text(json.dumps(meta, indent=2))
    return path

import numpy as np
import pytest

from csihar.parser import CsiParseError, LLTF_N_SUBCARRIERS, parse_line
from csihar.simulate import generate_lines


def make_line(n_values: int = 128) -> str:
    meta = (
        "CSI_DATA,7,aa:bb:cc:dd:ee:ff,-42,11,1,7,0,1,0,0,0,1,0,-92,0,6,0,"
        f"123456,0,128,1,{n_values},0"
    )
    data = ",".join(str((i % 20) - 10) for i in range(n_values))
    return f'{meta},"[{data}]"'


def test_parses_valid_line():
    frame = parse_line(make_line(), host_ts=100.5)
    assert frame is not None
    assert frame.seq == 7
    assert frame.mac == "aa:bb:cc:dd:ee:ff"
    assert frame.rssi == -42
    assert frame.noise_floor == -92
    assert frame.channel == 6
    assert frame.host_ts == 100.5
    assert frame.csi.shape == (64,)
    assert frame.csi.dtype == np.complex64


def test_imag_first_real_second():
    # bytes [3, 4, ...] -> first subcarrier = 4 + 3j
    values = [3, 4] + [0] * 126
    meta = (
        "CSI_DATA,0,aa:bb:cc:dd:ee:ff,-42,11,1,7,0,1,0,0,0,1,0,-92,0,6,0,"
        "1,0,128,1,128,0"
    )
    line = f'{meta},"[{",".join(map(str, values))}]"'
    frame = parse_line(line, host_ts=0.0)
    assert frame.csi[0] == pytest.approx(4 + 3j)


def test_non_csi_lines_return_none():
    assert parse_line("I (1234) wifi: connected", 0.0) is None
    assert parse_line("type,id,mac,rssi,rate,...", 0.0) is None
    assert parse_line("", 0.0) is None


@pytest.mark.parametrize(
    "mutation",
    [
        lambda l: l[:-2],                       # truncated array terminator
        lambda l: l.replace(',"[', ",["),       # missing quote
        lambda l: l.replace("-42", "oops"),     # non-integer metadata
        lambda l: l.replace(",128,0,\"[", ",127,0,\"["),  # len mismatch
    ],
)
def test_corrupt_lines_raise(mutation):
    with pytest.raises(CsiParseError):
        parse_line(mutation(make_line()), 0.0)


def test_amplitude_and_lltf_shapes():
    frame = parse_line(make_line(), 0.0)
    assert frame.lltf.shape == (LLTF_N_SUBCARRIERS,)
    assert frame.amplitude.shape == (LLTF_N_SUBCARRIERS,)
    assert (frame.amplitude >= 0).all()


def test_simulator_output_parses_end_to_end():
    lines = generate_lines("walking", duration_s=2.0, seed=1)
    frames = [parse_line(line, ts) for ts, line in lines]
    parsed = [f for f in frames if f is not None]
    assert len(parsed) > 150  # ~200 packets minus ~2% drops
    assert all(f.csi.shape == (64,) for f in parsed)

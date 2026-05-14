import json

import numpy as np

from edit_with_my_voice.core import append_edit, fit_to_len, safe_name, seconds_to_ts, ts_to_seconds


def test_timestamp_parsing():
    assert ts_to_seconds("01:02.500") == 62.5
    assert ts_to_seconds("01:02:03") == 3723
    assert seconds_to_ts(62.5) == "01:02.500"


def test_safe_name_removes_unsafe_chars():
    assert safe_name("../my voice!.wav") == "my_voice.wav"
    assert safe_name("!!!", default="x") == "x"


def test_append_edit_uses_generated_audio_when_manual_missing(tmp_path):
    generated = tmp_path / "generated.wav"
    generated.write_bytes(b"fake")
    out = append_edit("[]", "1", "2", None, str(generated), "demo")
    data = json.loads(out)
    assert data[0]["replacement"] == str(generated)
    assert data[0]["start"] == 1
    assert data[0]["end"] == 2


def test_append_edit_accepts_timestamp_json(tmp_path):
    generated = tmp_path / "generated.wav"
    generated.write_bytes(b"fake")
    existing = json.dumps([
        {"label": "demo", "start": "00:01.000", "end": "00:02.000", "replacement": str(generated)}
    ])
    out = append_edit(existing, "3", "4", None, str(generated), "second")
    data = json.loads(out)
    assert data[0]["start"] == 1
    assert data[0]["end"] == 2
    assert data[1]["start"] == 3


def test_fit_to_len_exact_length():
    y = np.linspace(-1, 1, 100, dtype=np.float32)
    out = fit_to_len(y, 50, preserve_pitch=False)
    assert len(out) == 50

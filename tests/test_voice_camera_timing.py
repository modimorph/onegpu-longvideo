import json

import numpy as np

from onegpu_longvideo import camera, voice
from onegpu_longvideo.timing import GpuSampler, TimingLog


def test_match_length():
    assert len(voice.match_length(np.ones(10), 7)) == 7
    y = voice.match_length(np.ones((5, 2)), 8)
    assert y.shape == (8, 2) and y[-1].sum() == 0


def test_envelope_lag():
    sr = 16000
    rng = np.random.default_rng(0)
    bursts = np.repeat(rng.random(200) > 0.6, sr // 50).astype(float)
    x = bursts * rng.normal(0, 1, len(bursts))
    y = np.r_[np.zeros(int(0.05 * sr)), x[: -int(0.05 * sr)]]
    lag, corr = voice.envelope_lag_ms(x, y, sr)
    assert abs(lag - 50) <= 10 and corr > 0.8


def test_wer_guard():
    assert voice.wer_guard([0, 0.1], [0, 0.1])
    assert not voice.wer_guard([0, 0], [0, 0.05])
    assert not voice.wer_guard([0], [0, 0])


def test_camera_classify_and_plan():
    assert camera.classify(1.1, 1) == "push-in"
    assert camera.classify(0.9, 0) == "pull-out"
    assert camera.classify(1.0, 10) == "arc/slide"
    assert camera.classify(1.0, 0.5) == "locked-off"
    r = {"move_detected": "push-in", "cuts": [], "jitter_px": 0.2}
    assert camera.matches_plan(r, "push") and not camera.matches_plan({**r, "cuts": [3.0]}, "push")
    assert camera.jitter([1.0] * 48, 24) < 1e-9


def test_timing_log(tmp_path):
    log = TimingLog(tmp_path / "t.jsonl")
    with log.stage("render", frames=719) as row:
        row["exec_s"] = 1.0
    rows = [json.loads(x) for x in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert rows[0]["stage"] == "render" and rows[0]["ok"] and rows[0]["exec_s"] == 1.0


def test_gpu_sampler_with_fake_probe():
    vals = iter([100, 300, 200] + [200] * 100)
    with GpuSampler(period=0.01, probe=lambda: next(vals)) as s:
        import time
        time.sleep(0.1)
    assert s.peak_mib == 300

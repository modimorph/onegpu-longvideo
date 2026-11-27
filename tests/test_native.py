"""The optional Rust cores (native/olv_native) must return exactly what the Python originals return. Skipped when not installed."""
import random

import numpy as np
import pytest

olv_native = pytest.importorskip("olv_native")

from onegpu_longvideo import voice  # noqa: E402
from onegpu_longvideo.qa import wer  # noqa: E402


def test_align_identical_on_random_pairs():
    rng = random.Random(7)
    vocab = [f"w{i}" for i in range(12)]
    for _ in range(1000):
        ref = [rng.choice(vocab) for _ in range(rng.randint(0, 25))]
        hyp = [rng.choice(vocab) for _ in range(rng.randint(0, 25))]
        assert [tuple(o) for o in olv_native.align(ref, hyp)] == wer.align_py(ref, hyp)


@pytest.mark.parametrize("n,sr", [(48000 * 30 + 17, 48000), (1000, 16000), (100, 48000), (480, 48000)])
def test_envelope_core_matches_numpy(n, sr):
    x = np.random.default_rng(n).standard_normal(n) * 0.3
    hop = int(sr * 0.01)
    a = voice.envelope(x, sr)
    if n < hop:  # np.convolve "same" pads to len(v); the Rust core is only for len(x) >= hop
        return
    b = np.asarray(olv_native.envelope(x.tolist(), hop))
    assert a.shape == b.shape
    assert np.max(np.abs(a - b)) < 1e-9

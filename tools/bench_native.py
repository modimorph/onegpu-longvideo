"""Per-part micro-bench: Rust cores vs the Python originals, same inputs, repeats inside ONE invocation, median [p20, p80] in ms.
usage: python tools/bench_native.py   (needs olv_native installed: pip install ./native/olv_native)"""
import json
import platform
import random
import timeit

import numpy as np
import olv_native
from onegpu_longvideo import voice
from onegpu_longvideo.qa import wer


def q(ts):
    ts = sorted(ts)
    return [round(ts[len(ts) // 2] * 1e3, 3), round(ts[int(len(ts) * 0.2)] * 1e3, 3), round(ts[int(len(ts) * 0.8)] * 1e3, 3)]


def bench(fn, n):
    fn()  # warm-up, excluded
    return q(timeit.repeat(fn, number=1, repeat=n))


rows = []
# envelope: 30 s of 48 kHz audio (the voice-post input size)
x = np.random.default_rng(0).standard_normal(48000 * 30) * 0.3
hop = 480


def cumsum_control():  # numpy prefix-sum control: is the gain from Rust or from the algorithm?
    c = np.concatenate(([0.0], np.cumsum(x * x / hop)))
    n = len(x)
    k = np.arange(0, n, hop) + (hop - 1) // 2
    hi = np.minimum(k, n - 1) + 1
    lo = np.maximum(k - hop + 1, 0)
    return np.sqrt(c[hi] - c[lo])


def rust_env():  # dropped from the package: loses to the numpy control
    return np.asarray(olv_native.envelope(x.tolist(), hop))


a, b, c = voice.envelope(x, 48000), rust_env(), cumsum_control()
rows.append({"part": "Q6n voice.envelope 30 s 48 kHz", "python_ms": bench(lambda: voice.envelope(x, 48000), 7),
             "numpy_cumsum_control_ms": bench(cumsum_control, 21), "rust_ms": bench(rust_env, 21),
             "max_abs_diff_rust": float(np.max(np.abs(a - b))), "max_abs_diff_cumsum": float(np.max(np.abs(a - c))), "tolerance": 1e-9})
# align: 30 s clip (about 90 words) and 3 min clip (about 540 words), about 10 % word errors
rng = random.Random(1)
vocab = [f"w{i}" for i in range(300)]
for label, nw in (("30 s, 90 words", 90), ("3 min, 540 words", 540)):
    ref = [rng.choice(vocab) for _ in range(nw)]
    hyp = [w if rng.random() > 0.1 else rng.choice(vocab) for w in ref]
    same = wer.align(ref, hyp) == wer.align_py(ref, hyp)
    rows.append({"part": f"Q5 qa.wer.align {label}", "python_ms": bench(lambda: wer.align_py(ref, hyp), 15),
                 "rust_ms": bench(lambda: wer.align(ref, hyp), 51), "identical_ops": same, "tolerance": "identical op list"})
for r in rows:
    r["speedup_vs_python"] = round(r["python_ms"][0] / r["rust_ms"][0], 1)
print(json.dumps({"machine": platform.platform(), "cpu": platform.processor(), "rows": rows}, indent=1))

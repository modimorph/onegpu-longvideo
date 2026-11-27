# Optional native cores

`olv_native` (Rust, PyO3 0.30, maturin 1.16): drop-in cores for pure-Python hot paths. The package works without it.

| Function | Python original | Result (Mac M2 Pro, median of repeats in one run) | Status |
|---|---|---|---|
| `align` | `qa.wer.align_py` | 30 s clip 2.30 -> 0.049 ms; 3 min clip 87.4 -> 1.02 ms; identical op lists on 1,000 random pairs | used by `qa.wer.align` when installed |
| `envelope` | `voice.envelope` | 91 -> 29 ms, but a numpy prefix-sum is 7.4 ms (the list copy into Rust costs more than the loop) | NOT used; a numpy fix is the better path |

Install: `pip install ./native/olv_native` (needs a Rust toolchain). Test: `pytest tests/test_native.py`. Bench: `python tools/bench_native.py`.

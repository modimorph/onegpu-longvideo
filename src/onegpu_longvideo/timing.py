"""Timing logger and GPU memory sampler. Every stage writes one JSON line: name, seconds, extra fields."""
from __future__ import annotations

import json
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path


class TimingLog:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.rows: list[dict] = []

    @contextmanager
    def stage(self, name: str, **extra):
        t0 = time.perf_counter()
        row = {"stage": name, **extra}
        try:
            yield row
            row["ok"] = True
        except BaseException as e:
            row["ok"] = False
            row["error"] = str(e)[:300]
            raise
        finally:
            row["seconds"] = round(time.perf_counter() - t0, 2)
            self.rows.append(row)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as f:
                    f.write(json.dumps(row) + "\n")

    def total(self) -> float:
        return round(sum(r["seconds"] for r in self.rows), 2)


def nvidia_smi_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.split()[0])
    except Exception:
        return None


class GpuSampler:
    """Samples total GPU memory every `period` seconds in a thread. On Windows (WDDM), also watch the
    shared-memory counter: a spill into shared memory made our renders several times slower."""

    def __init__(self, period: float = 5.0, probe=nvidia_smi_used_mib):
        self.period, self.probe = period, probe
        self.samples: list[tuple[float, int]] = []
        self._stop = threading.Event()
        self._t0 = time.time()

    def _run(self):
        while not self._stop.is_set():
            v = self.probe()
            if v is not None:
                self.samples.append((round(time.time() - self._t0, 1), v))
            self._stop.wait(self.period)

    def __enter__(self):
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._stop.set()

    @property
    def peak_mib(self) -> int | None:
        return max((v for _, v in self.samples), default=None)


def torch_profile(fn, out_dir: str | Path, steps: int = 1):
    """Run fn() under torch.profiler with CUDA activity and write a Chrome trace plus a top-30 kernel table."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(steps):
            fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(out / "trace.json"))
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
    (out / "top_kernels.txt").write_text(table)
    return table

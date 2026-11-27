"""Voice post: replace the generated voice with a converted one (for example Seed-VC) without breaking lip sync.

Rules from our runs (docs/EXPERIMENTS.md):
- the converted track must have EXACTLY the original sample count (pad or trim), or lip sync drifts;
- measure the envelope lag per window: it must stay within a few tens of milliseconds;
- keep the converted voice only if its per-window WER is not worse than the original (WER guard).
The Seed-VC call itself is a subprocess you configure (see configs/voice_post.example.yaml).
"""
from __future__ import annotations

import shlex
import subprocess

import numpy as np


def match_length(x: np.ndarray, n: int) -> np.ndarray:
    """Pad with zeros or trim so len(x) == n (sample-exact)."""
    x = np.asarray(x)
    if len(x) >= n:
        return x[:n]
    pad = [(0, n - len(x))] + [(0, 0)] * (x.ndim - 1)
    return np.pad(x, pad)


def envelope(x: np.ndarray, sr: int, hop_ms: float = 10.0) -> np.ndarray:
    hop = max(1, int(sr * hop_ms / 1000))
    x = np.asarray(x, dtype=np.float64)
    e = np.sqrt(np.convolve(x ** 2, np.ones(hop) / hop, "same"))
    return e[::hop]


def envelope_lag_ms(a: np.ndarray, b: np.ndarray, sr: int, max_lag_ms: float = 300.0, hop_ms: float = 10.0) -> tuple[float, float]:
    """Lag of b relative to a (positive = b later), by normalised cross-correlation of 10 ms RMS envelopes.
    Returns (lag_ms, correlation)."""
    e1, e2 = envelope(a, sr, hop_ms), envelope(b, sr, hop_ms)
    n = min(len(e1), len(e2))
    e1, e2 = e1[:n] - e1[:n].mean(), e2[:n] - e2[:n].mean()
    norm = np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-12
    maxl = int(max_lag_ms / hop_ms)
    best = (-2.0, 0)
    for lag in range(-maxl, maxl + 1):
        if lag >= 0:
            c = float(np.dot(e1[: n - lag], e2[lag:])) / norm
        else:
            c = float(np.dot(e1[-lag:], e2[: n + lag])) / norm
        if c > best[0]:
            best = (c, lag)
    return best[1] * hop_ms, best[0]


def windowed_lags(a: np.ndarray, b: np.ndarray, sr: int, win_s: float = 5.0) -> list[float]:
    out = []
    step = int(win_s * sr)
    for s in range(0, min(len(a), len(b)) - step + 1, step):
        out.append(envelope_lag_ms(a[s:s + step], b[s:s + step], sr, max_lag_ms=200.0)[0])
    return out


def wer_guard(win_wer_orig: list[float], win_wer_vc: list[float], tol: float = 0.0) -> bool:
    """Keep the converted voice only if no window got worse than the original by more than tol."""
    if len(win_wer_orig) != len(win_wer_vc):
        return False
    return all(v <= o + tol for o, v in zip(win_wer_orig, win_wer_vc))


def run_converter(cmd_template: str, source_wav: str, target_ref_wav: str, out_wav: str, timeout: int = 1800) -> None:
    """Run your voice converter. cmd_template uses {source}, {target}, {output}; for example
    'python -m seed_vc.inference --source {source} --target {target} --output {output}' (adapt to your install)."""
    cmd = cmd_template.format(source=shlex.quote(source_wav), target=shlex.quote(target_ref_wav), output=shlex.quote(out_wav))
    subprocess.run(cmd, shell=True, check=True, timeout=timeout)


def finalize(original_wav: str, converted_wav: str, out_wav: str, sr_out: int = 48000) -> dict:
    """Resample the converted track, make it sample-exact to the original, limit peaks, write stereo PCM16."""
    import soundfile as sf

    o, so = sf.read(original_wav, dtype="float32", always_2d=True)
    a, sa = sf.read(converted_wav, dtype="float32", always_2d=True)
    a = a.mean(1)
    if sa != so:
        import librosa

        a = librosa.resample(a, orig_sr=sa, target_sr=so)
    a = match_length(a, len(o))
    peak = float(np.max(np.abs(a))) if len(a) else 0.0
    if peak > 0.98:
        a = a * (0.98 / peak)
    sf.write(out_wav, np.stack([a] * o.shape[1], 1), so, subtype="PCM_16")
    lags = windowed_lags(o.mean(1), a, so)
    return {"samples": int(len(a)), "orig_samples": int(len(o)), "len_diff": int(len(a) - len(o)), "lags_ms": lags,
            "peak_before_limit": round(peak, 3)}

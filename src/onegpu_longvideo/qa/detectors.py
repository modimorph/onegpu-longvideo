"""Stage-1 detectors on per-frame series (pure numpy). Feature extraction (face embeddings, pose, SSIM) is separate,
so these functions are testable without a GPU or video.

- identity drift: cosine similarity of a face embedding to the reference, persistent steps and low runs;
- seams: frames whose difference to the previous frame is an outlier (a hidden cut or a chain join);
- gesture repeat: 2 s pose-descriptor segments that correlate highly with an earlier segment (a replayed gesture);
- AV offset: lag between mouth opening and audio envelope.
"""
from __future__ import annotations

import numpy as np


def runs(mask, gap: int = 0) -> list[tuple[int, int]]:
    """Inclusive index runs of True, merging gaps <= gap."""
    idx = np.where(np.asarray(mask, bool))[0]
    if len(idx) == 0:
        return []
    out = [[int(idx[0]), int(idx[0])]]
    for i in idx[1:]:
        if i - out[-1][1] <= gap + 1:
            out[-1][1] = int(i)
        else:
            out.append([int(i), int(i)])
    return [tuple(o) for o in out]


def stepdet(x, k: int = 12) -> np.ndarray:
    """Persistent step: |median(next k) - median(previous k)|. A one-frame flicker returns, a step stays."""
    x = np.asarray(x, float)
    n = len(x)
    out = np.zeros(n)
    for i in range(k, n - k):
        out[i] = abs(np.nanmedian(x[i:i + k]) - np.nanmedian(x[i - k:i]))
    return out


def identity_flags(cos_to_ref, fps: float, low: float = 0.5, step: float = 0.08, k: int = 12, min_len_s: float = 0.5) -> list[dict]:
    """Flag runs where identity similarity is low, and persistent steps (a face that changes and stays changed)."""
    c = np.asarray(cos_to_ref, float)
    flags = []
    for a, b in runs(np.nan_to_num(c, nan=1.0) < low, gap=int(fps * 0.25)):
        if (b - a + 1) / fps >= min_len_s:
            flags.append({"type": "identity_drift", "t_start": round(a / fps, 2), "t_end": round((b + 1) / fps, 2),
                          "value": round(float(np.nanmin(c[a:b + 1])), 3), "threshold": low})
    s = stepdet(c, k)
    for a, b in runs(s > step, gap=k):
        flags.append({"type": "identity_jump", "t_start": round(a / fps, 2), "t_end": round((b + 1) / fps, 2),
                      "value": round(float(s[a:b + 1].max()), 3), "threshold": step})
    return flags


def seam_flags(frame_diff, fps: float, z: float = 6.0) -> list[dict]:
    """Robust z-score of frame-to-frame difference; a spike far above the clip's own motion is a cut or seam."""
    d = np.asarray(frame_diff, float)
    med = np.nanmedian(d)
    mad = np.nanmedian(np.abs(d - med)) * 1.4826 + 1e-9
    zs = (d - med) / mad
    return [{"type": "seam", "t": round(i / fps, 2), "z": round(float(zs[i]), 1)} for i in np.where(zs > z)[0]]


def gesture_repeats(desc, fps: float, seg_s: float = 2.0, hop_s: float = 0.5, min_lag_s: float = 3.0,
                    r_min: float = 0.85, energy_min: float = 0.02) -> list[dict]:
    """desc: (n_frames, d) pose descriptor normalised by shoulder width. Segments get their linear trend removed
    (slow drift is not a gesture); low-energy segments (still hands) are ignored."""
    X = np.asarray(desc, float)
    n = len(X)
    seg, hop = int(seg_s * fps), max(1, int(hop_s * fps))
    starts = list(range(0, n - seg, hop))
    if len(starts) < 2:
        return []
    raw = []
    for s in starts:
        r = X[s:s + seg]
        x = np.arange(len(r)) - (len(r) - 1) / 2
        rc = r - r.mean(0)
        raw.append(rc - np.outer(x, (x @ rc) / (x @ x)))
    energy = np.array([float(np.mean(np.std(r, axis=0))) for r in raw])
    S = np.stack([r.ravel() for r in raw])
    S = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)
    C = S @ S.T / S.shape[1]
    st = np.array(starts) / fps
    out = []
    for a in range(len(st)):
        for b in range(a + 1, len(st)):
            if st[b] - st[a] >= min_lag_s and C[a, b] >= r_min and min(energy[a], energy[b]) >= energy_min:
                out.append({"type": "gesture_repeat", "t_first": round(float(st[a]), 2), "t_repeat": round(float(st[b]), 2),
                            "r": round(float(C[a, b]), 3)})
    return out


def av_offset_ms(mouth_open, audio_rms, fps: float, max_lag_frames: int = 8) -> tuple[float, float]:
    """Lag of the audio envelope vs mouth opening (both per video frame). Positive = audio later. Returns (ms, corr)."""
    m = np.nan_to_num(np.asarray(mouth_open, float))
    a = np.nan_to_num(np.asarray(audio_rms, float))
    n = min(len(m), len(a))
    m, a = m[:n] - m[:n].mean(), a[:n] - a[:n].mean()
    norm = np.linalg.norm(m) * np.linalg.norm(a) + 1e-12
    best = (-2.0, 0)
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        c = float(np.dot(m[: n - lag], a[lag:]) if lag >= 0 else np.dot(m[-lag:], a[: n + lag])) / norm
        if c > best[0]:
            best = (c, lag)
    return best[1] * 1000.0 / fps, best[0]


def merge_windows(flags: list[dict], pad_s: float = 0.5, max_len_s: float = 3.0) -> list[tuple[float, float]]:
    """Turn flags into stage-2 windows: pad, merge overlaps, split anything longer than max_len_s."""
    spans = []
    for f in flags:
        a = f.get("t_start", f.get("t", f.get("t_repeat", 0.0)))
        b = f.get("t_end", a)
        spans.append((max(0.0, a - pad_s), b + pad_s))
    spans.sort()
    merged: list[list[float]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out = []
    for a, b in merged:
        while b - a > max_len_s:
            out.append((round(a, 2), round(a + max_len_s, 2)))
            a += max_len_s
        out.append((round(a, 2), round(b, 2)))
    return out

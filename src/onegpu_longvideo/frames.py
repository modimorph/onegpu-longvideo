"""Frame, latent and token arithmetic for the MiniMax-H3 / FastH3 family (pure Python, no GPU).

The video VAE compresses time by 17 -> 5 latent steps after a 5-frame head, so valid lengths are 17k + 5 frames.
"""
from __future__ import annotations

import math

FPS = 24
AUDIO_ROWS_PER_SECOND = 80  # 2 audio rows per 40 Hz audio latent frame
ROPE_TIME_UNITS_PER_SECOND = 40.0
ROPE_THETA = 10000.0
ROPE_TIME_COMPONENTS = 16


def align_frames(n: int) -> int:
    """Smallest valid length >= n (valid = 17k + 5, minimum 5)."""
    n = max(5, int(n))
    return n + (5 - n % 17) % 17


def frames_for_seconds(seconds: float, fps: int = FPS) -> int:
    """Valid frame count that covers `seconds`. 29.96 s -> 719, 15 s -> 362, 10 s -> 243, 5 s -> 124."""
    return align_frames(round(seconds * fps))


def latent_t(frames: int) -> int:
    """Number of latent time steps for a valid frame count."""
    return 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2


def tokens(frames: int, width: int, height: int, text_ref_rows: int = 700) -> dict:
    """Estimate the packed sequence length: video rows (patch 1x2x2 on a /16 latent),
    audio rows and a text/reference allowance. Odd latent rows pad up."""
    lt = latent_t(frames)
    video = lt * (width // 32) * (((height // 16) + 1) // 2)
    audio = 2 * round(frames / FPS * 40)
    return {"latent_t": lt, "video_rows": video, "audio_rows": audio, "S": video + audio + text_ref_rows}


def rope_time_periods(theta: float = ROPE_THETA, n: int = ROPE_TIME_COMPONENTS,
                      units_per_second: float = ROPE_TIME_UNITS_PER_SECOND) -> list[float]:
    """Period in seconds of each temporal RoPE component.

    With theta 10000 and 40 units per second, component 8 has a period of about 15.7 s:
    the trained clip length. Gaps longer than that were never seen in training."""
    out = []
    for k in range(n):
        inv = theta ** (-k / n)
        out.append(2 * math.pi / inv / units_per_second)
    return out


def windows(duration_s: float, size_s: float = 5.0) -> list[tuple[float, float]]:
    """Split [0, duration) into fixed windows; the last window absorbs a short tail (< size/2)."""
    if duration_s <= 0:
        return []
    out = []
    t = 0.0
    while t < duration_s - 1e-9:
        out.append((t, min(t + size_s, duration_s)))
        t += size_s
    if len(out) > 1 and out[-1][1] - out[-1][0] < size_s / 2:
        a, _ = out.pop()
        out[-1] = (out[-1][0], duration_s)
    return out

"""Colour statistics and shot matching to an anchor grade (pixel math only, numpy + ffmpeg).

Method: Reinhard, Ashikhmin, Gooch, Shirley, "Color Transfer between Images" (IEEE CG&A, 2001): match per-channel mean and
standard deviation. We do it in CIE Lab (D65) instead of the paper's l-alpha-beta, bake it into a 33^3 .cube LUT per clip,
and apply it with ffmpeg `lut3d`. dE76 = Euclidean distance of Lab means (a coarse shot-to-shot drift number).
"""
from __future__ import annotations

import json
import subprocess

import numpy as np

W, H = 320, 180
_M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
_MI = np.linalg.inv(_M)
_WP = np.array([0.95047, 1.0, 1.08883])


def rgb2lab(rgb: np.ndarray) -> np.ndarray:
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = c @ _M.T / _WP
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab2rgb01(lab: np.ndarray) -> np.ndarray:
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f ** 3 > 216 / 24389, f ** 3, (116 * f - 16) / (24389 / 27)) * _WP
    c = xyz @ _MI.T
    c = np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)
    return np.clip(c, 0, 1)


def probe(path: str) -> dict:
    o = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,r_frame_rate,nb_frames:format=duration",
                        "-of", "json", path], capture_output=True, text=True, check=True).stdout
    d = json.loads(o)
    v = next((s for s in d["streams"] if s["codec_type"] == "video"), {})
    a, b = (v.get("r_frame_rate") or "24/1").split("/")
    return {"duration": float(d["format"]["duration"]), "width": v.get("width"), "height": v.get("height"),
            "fps": float(a) / float(b or 1), "has_audio": any(s["codec_type"] == "audio" for s in d["streams"])}


def frames_rgb(path: str, step: int = 4, start: float | None = None, end: float | None = None, crop: float = 1.0):
    """Yield RGB frames (H, W, 3) scaled to 320x180, every `step`-th frame of [start, end); crop > 1 = centre crop (punch-in)."""
    cmd = ["ffmpeg", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", path]
    if end is not None:
        cmd += ["-t", f"{max(0.04, end - (start or 0)):.3f}"]
    pre = f"crop=iw/{crop}:ih/{crop}," if crop and crop != 1.0 else ""
    cmd += ["-vf", f"{pre}scale={W}:{H}:flags=area:in_color_matrix=bt709:in_range=tv:out_range=full", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    i, n = 0, W * H * 3
    try:
        while True:
            b = p.stdout.read(n)
            if len(b) < n:
                break
            if i % step == 0:
                yield np.frombuffer(b, np.uint8).reshape(H, W, 3)
            i += 1
    finally:
        p.stdout.close()
        p.wait()


def lab_stats(frames: list[np.ndarray]) -> dict:
    if not frames:
        raise ValueError("no frames decoded")
    x = np.concatenate([rgb2lab(f).reshape(-1, 3)[::7] for f in frames])
    return {"mean": [round(float(v), 3) for v in x.mean(0)], "std": [round(float(v), 3) for v in x.std(0)]}


def lerp_stats(a: dict, b: dict, w: float) -> dict:
    """Stats between two keyframes (w = 0 -> a, 1 -> b)."""
    return {k: [x + w * (y - x) for x, y in zip(a[k], b[k])] for k in ("mean", "std")}


def de76(a: dict, b: dict) -> float:
    return round(float(np.linalg.norm(np.array(a["mean"]) - np.array(b["mean"]))), 2)


def write_cube(src: dict, ref: dict, path: str, n: int = 33, strength: float = 1.0) -> None:
    """3D LUT that moves src Lab statistics onto ref (Reinhard transfer). strength < 1 blends toward identity."""
    g = np.linspace(0, 1, n)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")          # .cube order: red changes fastest
    rgb01 = np.stack([r, gg, b], -1).reshape(-1, 3)
    lab = rgb2lab(rgb01 * 255.0)
    ms, ss, mr, sr = (np.array(v) for v in (src["mean"], src["std"], ref["mean"], ref["std"]))
    out = lab2rgb01((lab - ms) / np.maximum(ss, 1e-3) * sr + mr)
    out = strength * out + (1 - strength) * rgb01
    with open(path, "w") as f:
        f.write(f"LUT_3D_SIZE {n}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n")
        f.writelines(f"{x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in out)


LUT_CHAIN = ("scale=in_color_matrix=bt709:in_range=tv:out_range=full,format=gbrp,lut3d=file='{cube}':interp=tetrahedral,"
             "scale=in_range=full:out_color_matrix=bt709:out_range=tv,format=yuv420p")


def hard_cuts(path: str, start: float | None = None, end: float | None = None, ratio: float = 6.0, floor: float = 12.0) -> list[float]:
    """Times (s, from `start`) of hard cuts inside a clip: a frame-to-frame mean absolute luma change above
    max(floor, ratio x median change). A generated clip can switch camera by itself (seen: a set-up B clip that held the
    frontal view for 4.7 s, then cut)."""
    import numpy as np

    prev, diffs = None, []
    for f in frames_rgb(path, step=1, start=start, end=end):
        y = f.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        if prev is not None:
            diffs.append(float(np.abs(y - prev).mean()))
        prev = y
    if len(diffs) < 3:
        return []
    med = float(np.median(diffs))
    fps = probe(path)["fps"]
    thr = max(floor, ratio * med)
    return [round((i + 1) / fps, 3) for i, d in enumerate(diffs) if d > thr]


def frame_gray(path: str, t: float, width: int = 960):
    """One grey frame at time t (frame-accurate decode), or None."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", path, "-frames:v", "1", "-vf",
                          f"scale={width}:-2,format=gray", "-f", "rawvideo", "-"], capture_output=True).stdout
    if not out:
        return None
    h = len(out) // width
    return np.frombuffer(out[:h * width], np.uint8).reshape(h, width)


def framing_vs(ref, img) -> dict | None:
    """Similarity transform (scale, shift) from ref to img with ORB features + RANSAC (needs opencv). scale > 1 = closer."""
    import cv2

    orb = cv2.ORB_create(4000)
    ka, da = orb.detectAndCompute(ref, None)
    kb, db = orb.detectAndCompute(img, None)
    if da is None or db is None:
        return None
    m = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db), key=lambda x: x.distance)[:800]
    if len(m) < 20:
        return None
    pa = np.float32([ka[x.queryIdx].pt for x in m])
    pb = np.float32([kb[x.trainIdx].pt for x in m])
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=3)
    if M is None or inl.sum() < 20:
        return None
    return {"scale": round(float(np.hypot(M[0, 0], M[1, 0])), 3), "dx": round(float(M[0, 2]), 1), "dy": round(float(M[1, 2]), 1),
            "inliers": int(inl.sum())}

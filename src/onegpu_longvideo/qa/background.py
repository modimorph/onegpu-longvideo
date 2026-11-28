"""Background consistency check (review stage 1): is the room the same as in the reference?

Generated clips drift: a lamp moves, a second lamp appears, a window changes side. Per sampled frame we cut the person out
(MediaPipe selfie segmentation), keep the background both frames share, and compare it with the reference background:
  ssim   structural similarity of the background (grey, 480x270)       -> layout and texture changes
  edges  share of background edges that line up within 3 px (Canny)    -> new, moved or missing objects
  dE76   Lab mean difference of the background                         -> colour drift
  cell   the worst of a 6x4 grid of background cells (SSIM)            -> where it changed (a small object)
Runs of flagged samples become findings with a time range and a place. A vision model then looks only at those seconds
(review stage 2, `qa.stage2`). Needs numpy, opencv and mediapipe (`pip install opencv-python-headless mediapipe`).
"""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

W, H = 480, 270
GX, GY = 6, 4


def frames(path: str, fps: float = 2.0, start: float | None = None, end: float | None = None):
    """(t, RGB frame) at `fps`, scaled to 480x270."""
    cmd = ["ffmpeg", "-v", "error"] + (["-ss", f"{start:.3f}"] if start else []) + ["-i", path]
    if end is not None:
        cmd += ["-t", f"{end - (start or 0):.3f}"]
    still = path.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    # a still image has no frame rate: the fps filter returned no frame at all for a PNG reference (seen on the PC)
    if still:
        cmd += ["-vf", f"scale={W}:{H}:flags=area", "-frames:v", "1"]
    else:
        cmd += ["-vf", f"fps={fps},scale={W}:{H}:flags=area"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    i, n = 0, W * H * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield (start or 0) + i / fps, np.frombuffer(b, np.uint8).reshape(H, W, 3)
        i += 1
    p.wait()


class Segmenter:
    """Person mask (True = person), dilated so hair and motion blur stay out of the background."""

    def __init__(self, dilate_px: int = 9):
        import mediapipe as mp

        self.seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.k = dilate_px

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        import cv2

        m = self.seg.process(rgb).segmentation_mask > 0.5
        return cv2.dilate(m.astype(np.uint8), np.ones((self.k, self.k), np.uint8)).astype(bool)


def _ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    import cv2

    a, b = a.astype(np.float32), b.astype(np.float32)
    g = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)  # noqa: E731
    mu_a, mu_b = g(a), g(b)
    va, vb, cov = g(a * a) - mu_a ** 2, g(b * b) - mu_b ** 2, g(a * b) - mu_a * mu_b
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))


def compare(rgb: np.ndarray, ref: np.ndarray, bg: np.ndarray) -> dict:
    """Background metrics of one frame against the reference, on the shared background mask `bg`."""
    import cv2

    from ..colour import rgb2lab

    ga, gb = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    smap = _ssim_map(ga, gb)
    ssim = float(smap[bg].mean()) if bg.any() else 1.0
    ea, eb = cv2.Canny(ga, 60, 150) > 0, cv2.Canny(gb, 60, 150) > 0
    ea, eb = ea & bg, eb & bg
    k = np.ones((7, 7), np.uint8)
    da, db = cv2.dilate(ea.astype(np.uint8), k) > 0, cv2.dilate(eb.astype(np.uint8), k) > 0
    tot = ea.sum() + eb.sum()
    edges = float(((ea & db).sum() + (eb & da).sum()) / tot) if tot else 1.0
    la, lb = rgb2lab(rgb)[bg], rgb2lab(ref)[bg]
    de = float(np.linalg.norm(la.mean(0) - lb.mean(0))) if len(la) else 0.0
    cells = []
    for gy in range(GY):
        for gx in range(GX):
            sl = (slice(gy * H // GY, (gy + 1) * H // GY), slice(gx * W // GX, (gx + 1) * W // GX))
            m = bg[sl]
            if m.mean() > 0.5:                     # judge a cell only when most of it is background
                cells.append((float(smap[sl][m].mean()), gx, gy))
    worst = min(cells) if cells else (1.0, -1, -1)
    return {"ssim": round(ssim, 3), "edges": round(edges, 3), "dE76": round(de, 2),
            "cell_ssim": round(worst[0], 3), "cell": [worst[1], worst[2]], "bg_share": round(float(bg.mean()), 2)}


def where(cell: list[int]) -> str:
    gx, gy = cell
    if gx < 0:
        return "unknown"
    h = "left" if gx < GX / 3 else ("right" if gx >= 2 * GX / 3 else "centre")
    v = "top" if gy < GY / 3 else ("bottom" if gy >= 2 * GY / 3 else "middle")
    return f"{v} {h}"


def reference_frame(path: str, seg: Segmenter, seconds: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Reference background: an image, or the median of a video's first seconds; plus its person mask."""
    if path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        rgb = next(frames(path, fps=1))[1]
    else:
        fs = [f for _, f in frames(path, fps=4, end=seconds)]
        rgb = np.median(np.stack(fs), axis=0).astype(np.uint8)
    return rgb, seg(rgb)


def check(video: str, reference: str, fps: float = 2.0, th: dict | None = None, seg: Segmenter | None = None) -> dict:
    th = {"ssim": 0.55, "edges": 0.55, "dE76": 4.0, "cell_ssim": 0.30, "min_run": 2, "step": 0.15, **(th or {})}
    seg = seg or Segmenter()
    ref, ref_mask = reference_frame(reference, seg)
    rows = []
    for t, rgb in frames(video, fps=fps):
        bg = ~(seg(rgb) | ref_mask)
        m = compare(rgb, ref, bg)
        m["t"] = round(t, 2)
        m["flags"] = [k for k in ("ssim", "edges", "cell_ssim") if m[k] < th[k]] + (["dE76"] if m["dE76"] > th["dE76"] else [])
        rows.append(m)
    findings, run = [], []
    for r in rows + [{"flags": []}]:
        if r["flags"]:
            run.append(r)
            continue
        if len(run) >= th["min_run"]:
            worst = min(run, key=lambda x: x["cell_ssim"])
            findings.append({"t0": run[0]["t"], "t1": run[-1]["t"] + 1 / fps, "where": where(worst["cell"]),
                             "kinds": sorted({k for x in run for k in x["flags"]}),
                             "min_ssim": min(x["ssim"] for x in run), "min_edges": min(x["edges"] for x in run),
                             "max_dE76": max(x["dE76"] for x in run), "min_cell_ssim": worst["cell_ssim"]})
        run = []
    # a step = the room changes INSIDE the clip (seen: chained clips switch layout at about 4.7 s)
    for a, b in zip(rows, rows[1:]):
        if abs(b["ssim"] - a["ssim"]) > th["step"] or abs(b["cell_ssim"] - a["cell_ssim"]) > th["step"] * 2:
            findings.append({"t0": a["t"], "t1": b["t"], "where": where(b["cell"]), "kinds": ["step"],
                             "ssim_before": a["ssim"], "ssim_after": b["ssim"]})
    findings.sort(key=lambda f: f["t0"])
    summ = {k: (round(float(np.median([r[k] for r in rows])), 3) if rows else None) for k in ("ssim", "edges", "dE76", "cell_ssim")}
    return {"video": video, "reference": reference, "fps": fps, "thresholds": th, "median": summ,
            "findings": findings, "pass": not findings, "samples": rows}


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="background consistency vs a reference clip or image")
    ap.add_argument("video"); ap.add_argument("reference"); ap.add_argument("--fps", type=float, default=2.0); ap.add_argument("--out")
    a = ap.parse_args(argv)
    r = check(a.video, a.reference, fps=a.fps)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(r, f, indent=1)
    print(json.dumps({k: v for k, v in r.items() if k != "samples"}))
    sys.exit(0 if r["pass"] else 1)


if __name__ == "__main__":
    main()

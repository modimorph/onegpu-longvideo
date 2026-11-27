"""Camera-path check: estimate the camera move from background optical features and compare with the planned move.

Method: ORB features outside a central subject box, a similarity transform between consecutive frames (RANSAC),
accumulated into a path (scale, horizontal and vertical shift, roll). Jitter = std of the per-frame shift after a 1 s
moving average (shake). A frame with too few matches counts as a cut.
"""
from __future__ import annotations

import numpy as np

PLANS = {"push": "push-in", "pull": "pull-out", "arc": "arc/slide", "slide": "arc/slide", "locked": "locked-off"}


def classify(scale_end: float, pan_pct_w: float) -> str:
    zoom = scale_end - 1.0
    pan = abs(pan_pct_w)
    if zoom > 0.05 and pan < 6:
        return "push-in"
    if zoom < -0.05:
        return "pull-out"
    if pan >= 3:
        return "arc/slide"
    return "locked-off"


def jitter(dx: list[float], fps: float) -> float:
    if not dx:
        return 0.0
    d = np.asarray(dx, float)
    k = max(1, int(fps))
    pad = np.pad(d, (k // 2, k - 1 - k // 2), mode="edge")
    sm = np.convolve(pad, np.ones(k) / k, mode="valid")
    return float(np.std(d - sm))


def matches_plan(result: dict, plan: str, max_jitter_px: float = 1.0) -> bool:
    return result["move_detected"] == PLANS[plan] and not result["cuts"] and result["jitter_px"] < max_jitter_px


def analyze(video: str, subject_box=(0.28, 0.72), plan: str | None = None) -> dict:
    import cv2

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    orb = cv2.ORB_create(1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    prev = None
    S, X, Y, R, cuts, dX = [1.0], [0.0], [0.0], [0.0], [], []
    i, w, h = 0, 1, 1
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        h, w = g.shape
        mask = np.full_like(g, 255)
        mask[int(0.05 * h):h, int(subject_box[0] * w):int(subject_box[1] * w)] = 0
        kp, des = orb.detectAndCompute(g, mask)
        if prev is not None and des is not None and prev[1] is not None and len(kp) > 20:
            mt = sorted(bf.match(prev[1], des), key=lambda z: z.distance)[:300]
            A, inl = (None, None)
            if len(mt) >= 15:
                A, inl = cv2.estimateAffinePartial2D(np.float32([prev[0][z.queryIdx].pt for z in mt]),
                                                     np.float32([kp[z.trainIdx].pt for z in mt]), method=cv2.RANSAC)
            if A is None or (inl is not None and inl.sum() < 10):
                cuts.append(round(i / fps, 2))
                ds, dx, dy, dr = 1.0, 0.0, 0.0, 0.0
            else:
                ds = float(np.hypot(A[0, 0], A[0, 1]))
                dx, dy = float(A[0, 2]), float(A[1, 2])
                dr = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
            S.append(S[-1] * ds); X.append(X[-1] + dx); Y.append(Y[-1] + dy); R.append(R[-1] + dr); dX.append(dx)
        prev = (kp, des)
        i += 1
    res = {"frames": len(S), "duration_s": round(len(S) / fps, 2), "scale_end": round(S[-1], 3), "pan_pct_w": round(100 * X[-1] / w, 1),
           "tilt_pct_h": round(100 * Y[-1] / h, 1), "roll_deg": round(R[-1], 2), "jitter_px": round(jitter(dX, fps), 3), "cuts": cuts[:10]}
    res["move_detected"] = classify(res["scale_end"], res["pan_pct_w"])
    if plan:
        res["plan"] = PLANS[plan]
        res["match"] = matches_plan(res, plan)
    return res

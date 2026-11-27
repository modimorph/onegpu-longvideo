"""Upscale I/O lever bench: one invocation, one warm-up (excluded), one timed run per lever, plus a micro-profile.

Usage: python tools/bench_upscale_io.py IN.mp4 MODEL.safetensors OUT_DIR [lever,lever ...]
Gates (fixed before the run): bit-exact levers (overlap, hwdec, triton_post) need max abs diff 0 on every frame vs
the baseline run; non-exact levers (cudnn_bench, channels_last, interp_half) need PSNR >= 50 dB and max abs <= 3.
Prints one JSON line per run and a final summary JSON.
"""
import json
import math
import os
import sys
import time

T = {}
t = time.perf_counter(); import numpy as np; import torch; T["import_torch_s"] = time.perf_counter() - t
t = time.perf_counter(); torch.zeros(1).cuda(); torch.cuda.synchronize(); T["cuda_init_s"] = time.perf_counter() - t
t = time.perf_counter(); import spandrel; T["import_spandrel_s"] = time.perf_counter() - t
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from onegpu_longvideo import upscale as U  # noqa: E402

inp, mp, od = sys.argv[1], sys.argv[2], sys.argv[3]
levers = sys.argv[4].split(",") if len(sys.argv) > 4 else ["overlap", "hwdec", "cudnn_bench", "channels_last", "interp_half"]
EXACT = {"overlap", "hwdec", "triton_post", "ts_cache"}
os.makedirs(od, exist_ok=True)
W, H = 1920, 1080


def gpu_state():
    import subprocess
    q = "temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used"
    return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


def run(name, opts, ref=None):
    torch.backends.cudnn.benchmark = False
    frames, stats = [], {"maxabs": 0, "se": 0.0, "n": 0}

    def tap(a):
        if ref is None:
            frames.append(a.copy())
            return
        for f in a:
            if stats["n"] >= len(ref):
                stats["n"] += 1  # extra frame: fails the count check below
                continue
            r = ref[stats["n"]]
            d = np.abs(f.astype(np.int16) - r.astype(np.int16))
            stats["maxabs"] = max(stats["maxabs"], int(d.max()))
            stats["se"] += float((d.astype(np.float64) ** 2).mean())
            stats["n"] += 1
    full = {k: False for k in U.OPTS}
    full.update(opts)
    t0 = time.perf_counter()
    r = U.upscale(inp, os.path.join(od, f"{name}.mp4"), mp, W, H, opts=full, tap=tap)
    r["wall_s"] = round(time.perf_counter() - t0, 2)
    r["name"] = name
    r["gpu"] = gpu_state()
    if ref is not None:
        mse = stats["se"] / max(stats["n"], 1)
        r["maxabs"] = stats["maxabs"]
        r["psnr"] = "inf" if mse == 0 else round(10 * math.log10(255 ** 2 / mse), 2)
        r["frames_cmp"] = stats["n"]
        exact = all(k in EXACT for k in opts if opts[k])
        r["gate"] = "FAIL" if stats["n"] != len(ref) or not ref else "pass" if (stats["maxabs"] == 0 if exact else (mse == 0 or (10 * math.log10(255 ** 2 / mse) >= 50 and stats["maxabs"] <= 3))) else "FAIL"
    print(json.dumps(r), flush=True)
    return r, frames


def micro():
    """Per-component GPU time on one real batch of 8 frames (CUDA events, median of 20)."""
    import subprocess
    import torch.nn.functional as F
    pr = U.probe(inp)
    w, h = pr["width"], pr["height"]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", inp, "-frames:v", "8", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    x = torch.from_numpy(np.frombuffer(raw, np.uint8).reshape(8, h, w, 3).copy())
    m = spandrel.ModelLoader().load_from_file(mp).model.eval().cuda().half()
    out = {}

    def bench(name, fn, reps=20):
        fn(); torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); fn(); b.record(); b.synchronize()
            ts.append(a.elapsed_time(b))
        ts.sort()
        out[name] = [round(ts[len(ts) // 2], 3), round(ts[len(ts) // 5], 3), round(ts[4 * len(ts) // 5], 3)]

    with torch.inference_mode():
        bench("h2d_pageable_ms", lambda: x.cuda())
        xp = x.pin_memory()
        bench("h2d_pinned_ms", lambda: xp.cuda(non_blocking=True))
        t = x.cuda().permute(0, 3, 1, 2).half().div_(255)
        bench("model_fp16_ms", lambda: m(t))
        y = m(t)
        bench("interp_fp32_ms", lambda: F.interpolate(y.float(), size=(H, W), mode="bicubic", antialias=True))
        bench("interp_fp16_ms", lambda: F.interpolate(y, size=(H, W), mode="bicubic", antialias=True))
        z = F.interpolate(y.float(), size=(H, W), mode="bicubic", antialias=True)
        bench("post_torch_ms", lambda: z.clone().clamp_(0, 1).mul_(255).round_().byte().permute(0, 2, 3, 1).contiguous())
        ob = z.clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).contiguous()
        bench("d2h_pageable_ms", lambda: ob.cpu())
        pin = torch.empty_like(ob, device="cpu").pin_memory()
        bench("d2h_pinned_ms", lambda: pin.copy_(ob, non_blocking=True))
        torch.backends.cudnn.benchmark = True
        bench("model_fp16_cudnnbench_ms", lambda: m(t))
        mc = m.to(memory_format=torch.channels_last)
        tc = t.contiguous(memory_format=torch.channels_last)
        bench("model_fp16_cl_cudnnbench_ms", lambda: mc(tc))
        torch.backends.cudnn.benchmark = False
    try:
        t0 = time.perf_counter()
        from onegpu_longvideo.kernels.frame_io_triton import f32_nchw_to_u8_nhwc
        out["import_triton_s"] = round(time.perf_counter() - t0, 3)
        t0 = time.perf_counter(); f32_nchw_to_u8_nhwc(z.contiguous()); torch.cuda.synchronize()
        out["triton_first_call_s"] = round(time.perf_counter() - t0, 3)
        bench("post_triton_ms", lambda: f32_nchw_to_u8_nhwc(z))
    except Exception as e:  # noqa: BLE001
        out["triton_error"] = repr(e)[:200]
    return out


print(json.dumps({"startup": {k: round(v, 3) for k, v in T.items()}, "gpu": gpu_state(), "torch": torch.__version__}), flush=True)
t0 = time.perf_counter(); U.probe(inp); print(json.dumps({"ffprobe_s": round(time.perf_counter() - t0, 3)}), flush=True)
mode = os.environ.get("BENCH_MODE", "all")
if mode in ("all", "micro"):
    print(json.dumps({"micro": micro()}), flush=True)
if mode in ("all", "runs"):
    run("warmup", {})  # excluded
    base, ref = run("baseline", {})
    res = [base]
    for lv in levers:
        res.append(run(lv, {k: True for k in lv.split("+")}, ref)[0])
    print(json.dumps({"summary": [{k: r.get(k) for k in ("name", "pipeline_s", "gpu_s", "wall_s", "psnr", "maxabs", "gate")} for r in res]}), flush=True)

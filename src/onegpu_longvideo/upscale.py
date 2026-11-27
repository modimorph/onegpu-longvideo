"""Streaming x2 video upscaler (SPAN or any spandrel-loadable model) with overlapped decode, GPU and encode.

ffmpeg decode (rgb24 pipe) -> model (fp16, batches) -> GPU bicubic resize to the target size -> ffmpeg encode,
with the source audio stream copied untouched. Three threads overlap, so RAM stays small: no full-clip tensor.
Needs: torch with CUDA, spandrel, ffmpeg on PATH. Encoder: h264_nvenc (default) or libx264 (--x264).
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time


def probe(path: str) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
                          "-of", "json", path], capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"][0]


def encoder_args(x264: bool, quality: int) -> list[str]:
    if x264:
        return ["-c:v", "libx264", "-crf", str(quality), "-preset", "medium"]
    return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(quality), "-b:v", "0"]


def upscale(inp: str, out: str, model_path: str, width: int, height: int, batch: int = 8, quality: int = 18,
            x264: bool = False, comment: str = "AI-generated") -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from spandrel import ModelLoader

    t0 = time.perf_counter()
    pr = probe(inp)
    w, h, fps = pr["width"], pr["height"], pr["r_frame_rate"]
    model = ModelLoader().load_from_file(model_path).model.eval().cuda().half()
    t_load = time.perf_counter() - t0
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", inp, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", fps, "-i", "-",
                            "-i", inp, "-map", "0:v", "-map", "1:a?", *encoder_args(x264, quality), "-pix_fmt", "yuv420p", "-c:a", "copy",
                            "-map_metadata", "-1", "-metadata", f"comment={comment}", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    qin: queue.Queue = queue.Queue(4)
    qout: queue.Queue = queue.Queue(4)
    fsz = w * h * 3

    def reader():
        while True:
            buf = []
            for _ in range(batch):
                b = dec.stdout.read(fsz)
                if len(b) < fsz:
                    break
                buf.append(np.frombuffer(b, np.uint8).reshape(h, w, 3))
            if buf:
                qin.put(np.stack(buf))
            if len(buf) < batch:
                qin.put(None)
                return

    def writer():
        while True:
            x = qout.get()
            if x is None:
                break
            enc.stdin.write(x.tobytes())
        enc.stdin.close()

    threading.Thread(target=reader, daemon=True).start()
    wt = threading.Thread(target=writer)
    wt.start()
    post = None
    if os.environ.get("ONEGPU_TRITON_FRAMEIO") == "1":  # opt-in until the end-to-end A/B passes
        from .kernels.frame_io_triton import f32_nchw_to_u8_nhwc as post
    t1 = time.perf_counter()
    gpu = 0.0
    n = 0
    with torch.inference_mode():
        while True:
            x = qin.get()
            if x is None:
                break
            g0 = time.perf_counter()
            t = torch.from_numpy(x).cuda().permute(0, 3, 1, 2).half().div_(255)
            y = model(t)
            y = F.interpolate(y.float(), size=(height, width), mode="bicubic", antialias=True)
            if post is None:
                y = y.clamp_(0, 1)
            if post is not None:  # fused Triton: clamp*255, rint, byte, NHWC in one pass (bit-exact)
                o = post(y.contiguous()).cpu().numpy()
            else:
                o = y.mul_(255).round_().byte().permute(0, 2, 3, 1).contiguous().cpu().numpy()
            torch.cuda.synchronize()
            gpu += time.perf_counter() - g0
            qout.put(o)
            n += len(x)
    qout.put(None)
    wt.join()
    enc.wait()
    dec.wait()
    t2 = time.perf_counter()
    return {"in": inp, "out": out, "size": [width, height], "frames": n, "load_s": round(t_load, 1), "pipeline_s": round(t2 - t1, 1),
            "gpu_s": round(gpu, 1), "fps": round(n / max(t2 - t1, 1e-9), 2), "encoder": "x264" if x264 else "nvenc",
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}

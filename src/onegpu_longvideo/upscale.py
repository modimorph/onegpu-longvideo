"""Streaming x2 video upscaler (SPAN or any spandrel-loadable model) with overlapped decode, GPU and encode.

ffmpeg decode (rgb24 pipe) -> model (fp16, batches) -> GPU bicubic resize to the target size -> ffmpeg encode,
with the source audio stream copied untouched. Three threads overlap, so RAM stays small: no full-clip tensor.
Needs: torch with CUDA, spandrel, ffmpeg on PATH. Encoder: h264_nvenc (default) or libx264 (--x264).

Optional levers (each measured on its own; see onegpu-native/iterations/io-details.md). Defaults keep the
original path. Pass `opts` or set ONEGPU_UPSCALE_OPTS="cudnn_bench,channels_last,..." to switch them on:
  cudnn_bench   torch.backends.cudnn.benchmark = True (autotuned conv algorithms; not bit-exact)
  channels_last model and input in NHWC memory format (tensor-core friendly convs)
  overlap       pinned host buffers + a copy stream: D2H of batch k runs while batch k+1 computes; no
                device-wide synchronize per batch
  interp_half   bicubic resize in fp16 instead of fp32 (not bit-exact)
  hwdec         ffmpeg decode with -hwaccel cuda (NVDEC)
  triton_post   fused Triton post-process (old opt-in, also ONEGPU_TRITON_FRAMEIO=1)
  ts_cache      load a cached TorchScript trace of the fp16 model instead of importing spandrel (spandrel's
                registry pulls torchvision: ~2 s per process on the PC). First use traces and saves it.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time

OPTS = ("cudnn_bench", "channels_last", "overlap", "interp_half", "hwdec", "triton_post", "ts_cache")


def _ts_path(model_path: str) -> str:
    import hashlib

    import torch
    st = os.stat(model_path)
    key = hashlib.sha256(f"{os.path.abspath(model_path)}|{st.st_size}|{st.st_mtime_ns}|{torch.__version__}".encode()).hexdigest()[:16]
    d = os.environ.get("ONEGPU_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "onegpu-longvideo")
    return os.path.join(d, f"{os.path.basename(model_path)}.{key}.ts.pt")


class _Plain:
    """Run a TorchScript module without the JIT fusers, so it calls the same ATen kernels as eager (bit-exact)."""

    def __init__(self, m):
        self.m = m

    def to(self, *a, **k):
        self.m = self.m.to(*a, **k)
        return self

    def __call__(self, x):
        import torch
        with torch.jit.optimized_execution(False):
            return self.m(x)


def load_model(model_path: str, ts_cache: bool = False):
    """fp16 CUDA model. ts_cache: TorchScript trace cached per (file, size, mtime, torch version)."""
    import torch
    if ts_cache:
        p = _ts_path(model_path)
        if os.path.exists(p):
            return _Plain(torch.jit.load(p, map_location="cuda").eval())
    from spandrel import ModelLoader
    m = ModelLoader().load_from_file(model_path).model.eval().cuda().half()
    if ts_cache:
        with torch.inference_mode(False), torch.no_grad():
            tr = torch.jit.trace(m, torch.rand(1, 3, 64, 64, device="cuda", dtype=torch.float16), check_trace=False)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = f"{p}.{os.getpid()}.{threading.get_ident()}.tmp"  # unique per writer; os.replace is atomic
        tr.save(tmp)
        os.replace(tmp, p)
    return m


def probe(path: str) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
                          "-of", "json", path], capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"][0]


def encoder_args(x264: bool, quality: int) -> list[str]:
    if x264:
        return ["-c:v", "libx264", "-crf", str(quality), "-preset", "medium"]
    return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(quality), "-b:v", "0"]


def resolve_opts(opts: dict | None) -> dict:
    env = {k.strip() for k in os.environ.get("ONEGPU_UPSCALE_OPTS", "").split(",") if k.strip()}
    if env - set(OPTS):
        raise ValueError(f"unknown ONEGPU_UPSCALE_OPTS: {sorted(env - set(OPTS))}")
    o = {k: (k in env) for k in OPTS}
    if os.environ.get("ONEGPU_TRITON_FRAMEIO") == "1":
        o["triton_post"] = True
    o.update(opts or {})
    unknown = set(o) - set(OPTS)
    if unknown:
        raise ValueError(f"unknown upscale opts: {sorted(unknown)}")
    return o


def upscale(inp: str, out: str, model_path: str, width: int, height: int, batch: int = 8, quality: int = 18,
            x264: bool = False, comment: str = "AI-generated", opts: dict | None = None, model=None, tap=None) -> dict:
    """tap: optional callable(np.ndarray uint8 NHWC) that sees every output batch (bench correctness gate)."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    o = resolve_opts(opts)
    t0 = time.perf_counter()
    pr = probe(inp)
    w, h, fps = pr["width"], pr["height"], pr["r_frame_rate"]
    cudnn_prev = torch.backends.cudnn.benchmark
    if o["cudnn_bench"]:
        torch.backends.cudnn.benchmark = True
    try:
        if model is None:
            model = load_model(model_path, o["ts_cache"])
    except Exception:
        torch.backends.cudnn.benchmark = cudnn_prev
        raise
    if o["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    post = None
    if o["triton_post"]:
        from .kernels.frame_io_triton import f32_nchw_to_u8_nhwc as post
    t_load = time.perf_counter() - t0
    dec_cmd = ["ffmpeg", "-v", "error"] + (["-hwaccel", "cuda"] if o["hwdec"] else []) + ["-i", inp, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", fps, "-i", "-",
                            "-i", inp, "-map", "0:v", "-map", "1:a?", *encoder_args(x264, quality), "-pix_fmt", "yuv420p", "-c:a", "copy",
                            "-map_metadata", "-1", "-metadata", f"comment={comment}", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    qin: queue.Queue = queue.Queue(4)
    qout: queue.Queue = queue.Queue(4)
    fsz = w * h * 3

    stop = threading.Event()

    def put_in(v) -> bool:  # bounded put that gives up once the consumer has stopped
        while not stop.is_set():
            try:
                qin.put(v, timeout=0.5)
                return True
            except queue.Full:
                pass
        return False

    rerr: list = []
    reof = threading.Event()

    def reader():
        try:
            while True:
                buf = []
                for _ in range(batch):
                    b = dec.stdout.read(fsz)
                    if len(b) < fsz:
                        reof.set()  # clean end of the decoder's stream: its exit status decides success
                        break
                    buf.append(np.frombuffer(b, np.uint8).reshape(h, w, 3))
                if buf and not put_in(np.stack(buf)):
                    return
                if len(buf) < batch:
                    return
        except Exception as e:  # noqa: BLE001  (surfaced after teardown, so it never looks like a clean EOF)
            rerr.append(e)
        finally:
            put_in(None)  # also on a reader exception, so the GPU loop never waits forever

    werr: list = []

    def writer():
        # On any error: remember it, keep draining qout and returning pinned slots so the producer never blocks.
        while True:
            x = qout.get()
            if x is None:
                break
            try:
                if isinstance(x, tuple):  # (event, pinned tensor, n, slot release): wait for its async D2H copy
                    ev, pin, k, release = x
                    try:
                        ev.synchronize()  # also on the error path: the slot may be reused only after its copy finished
                        if not werr:
                            a = pin[:k].numpy()
                            if tap is not None:
                                tap(a.copy())  # the pinned slot is reused by a later batch
                            enc.stdin.write(a.tobytes())
                    finally:
                        release.put(pin)
                elif not werr:
                    if tap is not None:
                        tap(x)
                    enc.stdin.write(x.tobytes())
            except Exception as e:  # noqa: BLE001
                werr.append(e)
        try:
            enc.stdin.close()
        except OSError as e:
            werr.append(e)

    copy_stream = None
    pins: queue.Queue = queue.Queue()
    if o["overlap"]:  # allocated before the workers start, so a failed allocation leaves no thread blocked
        try:
            copy_stream = torch.cuda.Stream()
            for _ in range(3):  # 3 pinned slots: compute k+1 and k+2 while the writer drains k
                pins.put(torch.empty((batch, height, width, 3), dtype=torch.uint8, pin_memory=True))
        except Exception:  # stop and reap both ffmpeg children before re-raising
            for p_ in (dec, enc):
                if p_.poll() is None:
                    p_.kill()
                p_.wait()
            torch.backends.cudnn.benchmark = cudnn_prev
            raise
    threading.Thread(target=reader, daemon=True).start()
    wt = threading.Thread(target=writer)
    wt.start()
    t1 = time.perf_counter()
    gpu = 0.0
    n = 0
    try:
        with torch.inference_mode():
            while True:
                x = qin.get()
                if x is None:
                    break
                g0 = time.perf_counter()
                xt = torch.from_numpy(x)
                if o["overlap"]:
                    xt = xt.pin_memory().cuda(non_blocking=True)
                else:
                    xt = xt.cuda()
                t = xt.permute(0, 3, 1, 2).half().div_(255)
                if o["channels_last"]:
                    t = t.contiguous(memory_format=torch.channels_last)
                y = model(t)
                if o["interp_half"]:
                    y = F.interpolate(y, size=(height, width), mode="bicubic", antialias=True).float()
                else:
                    y = F.interpolate(y.float(), size=(height, width), mode="bicubic", antialias=True)
                if post is not None:  # fused Triton: clamp*255, rint, byte, NHWC in one pass (bit-exact)
                    ob = post(y.contiguous())
                else:
                    ob = y.clamp_(0, 1).mul_(255).round_().byte().permute(0, 2, 3, 1).contiguous()
                if o["overlap"]:
                    pin = pins.get()
                    done = torch.cuda.current_stream().record_event()
                    with torch.cuda.stream(copy_stream):
                        copy_stream.wait_event(done)
                        pin[: len(x)].copy_(ob, non_blocking=True)
                        ob.record_stream(copy_stream)
                        ev = copy_stream.record_event()
                    gpu += time.perf_counter() - g0
                    qout.put((ev, pin, len(x), pins))
                else:
                    a = ob.cpu().numpy()
                    torch.cuda.synchronize()
                    gpu += time.perf_counter() - g0
                    qout.put(a)
                n += len(x)
    finally:  # always stop the writer and the ffmpeg processes and restore cuDNN, also when the GPU loop raises
        stop.set()
        qout.put(None)
        wt.join()
        enc.wait()
        dec_killed = False
        dec_hung = False
        if reof.is_set():  # normal end: wait for the decoder's own exit status instead of killing it
            try:
                dec.wait(timeout=30)
            except subprocess.TimeoutExpired:
                dec_hung = True  # closed its stream but did not exit: treated as a failure below
        else:
            dec_killed = dec.poll() is None  # cancellation (error or early stop): kill is expected, status ignored
        if dec_killed or dec_hung or dec.poll() is None:
            dec.kill()
        dec.wait()
        torch.backends.cudnn.benchmark = cudnn_prev
    if werr:
        raise RuntimeError(f"upscale writer failed: {werr[0]!r}") from werr[0]
    if rerr:
        raise RuntimeError(f"upscale reader failed: {rerr[0]!r}") from rerr[0]
    if dec_hung or (dec.returncode and not dec_killed):
        raise RuntimeError(f"upscale decoder exited with code {dec.returncode}: input {inp} was not fully read")
    if enc.returncode:
        raise RuntimeError(f"upscale encoder exited with code {enc.returncode}: output {out} is not valid")
    t2 = time.perf_counter()
    return {"in": inp, "out": out, "size": [width, height], "frames": n, "load_s": round(t_load, 1), "pipeline_s": round(t2 - t1, 2),
            "gpu_s": round(gpu, 2), "fps": round(n / max(t2 - t1, 1e-9), 2), "encoder": "x264" if x264 else "nvenc",
            "opts": sorted(k for k, v in o.items() if v), "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}

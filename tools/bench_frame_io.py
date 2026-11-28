"""Smoke test + reference check + micro-bench for kernels/frame_io_triton.py.

Gates written before the first run (27 Nov 2026):
  pre : max abs error vs PyTorch <= 1e-3 (1 fp16 ulp near 1.0)
  post: max abs error vs PyTorch <= 0 (uint8, bit-exact expected)
Bench: triton.testing.do_bench median [p20, p80] in ONE invocation; GPU work only (H2D/D2H excluded, same in both paths).
Shapes = upscale() defaults: batch 8, 768x432 in, SPAN x2 -> 1536x864, bicubic to 1920x1080 out.
Usage: python tools/bench_frame_io.py   (needs CUDA + triton; set PYTHONPATH to src)
"""
import json
import sys
import time

import torch
import triton
from triton.testing import do_bench

from onegpu_longvideo.kernels.frame_io_triton import f32_nchw_to_u8_nhwc, u8_nhwc_to_f16_nchw


def smoke():
    import triton.language as tl

    @triton.jit
    def add(a, b, o, n, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        m = i < n
        tl.store(o + i, tl.load(a + i, mask=m) + tl.load(b + i, mask=m), mask=m)

    a = torch.rand(1 << 20, device="cuda"); b = torch.rand_like(a); o = torch.empty_like(a)
    add[(triton.cdiv(a.numel(), 1024),)](a, b, o, a.numel(), B=1024)
    return float((o - (a + b)).abs().max())


def ref_pre(x):
    return x.permute(0, 3, 1, 2).half().div_(255)


def ref_post(y):
    return y.clamp(0, 1).mul_(255).round_().byte().permute(0, 2, 3, 1).contiguous()


def main():
    torch.manual_seed(7329)
    dev = torch.cuda.get_device_name(0)
    res = {"device": dev, "torch": torch.__version__, "triton": triton.__version__}
    res["smoke_vector_add_maxabs"] = smoke()
    # realistic inputs: random pixels, and an upscaler output that over/undershoots [0,1]
    x = torch.randint(0, 256, (8, 432, 768, 3), dtype=torch.uint8, device="cuda")
    y = (torch.rand(8, 3, 1080, 1920, device="cuda") * 1.2 - 0.1)
    y[0, :, :4, :4] = torch.tensor([0.5 / 255, 1.5 / 255, 2.5 / 255, 254.5 / 255], device="cuda")  # ties
    for _ in range(2):  # second seed of inputs for the correctness check
        a, b = ref_pre(x), u8_nhwc_to_f16_nchw(x)
        res.setdefault("pre_maxabs", []).append(float((a.float() - b.float()).abs().max()))
        a, b = ref_post(y), f32_nchw_to_u8_nhwc(y)
        res.setdefault("post_maxabs", []).append(int((a.int() - b.int()).abs().max()))
        res.setdefault("post_mismatch_frac", []).append(float((a != b).float().mean()))
        x = torch.randint(0, 256, x.shape, dtype=torch.uint8, device="cuda")
        y = torch.randn(y.shape, device="cuda") * 0.3 + 0.5
    res["pre_pass"] = max(res["pre_maxabs"]) <= 1e-3
    res["post_pass"] = max(res["post_maxabs"]) <= 0
    q = [0.5, 0.2, 0.8]
    res["pre_torch_ms"] = do_bench(lambda: ref_pre(x), quantiles=q, rep=300)
    res["pre_triton_ms"] = do_bench(lambda: u8_nhwc_to_f16_nchw(x), quantiles=q, rep=300)
    res["post_torch_ms"] = do_bench(lambda: ref_post(y), quantiles=q, rep=300)
    res["post_triton_ms"] = do_bench(lambda: f32_nchw_to_u8_nhwc(y), quantiles=q, rep=300)
    # context: the bicubic step that sits between them in upscale()
    import torch.nn.functional as F
    z = torch.rand(8, 3, 864, 1536, device="cuda", dtype=torch.float16)
    res["bicubic_ctx_ms"] = do_bench(lambda: F.interpolate(z.float(), size=(1080, 1920), mode="bicubic", antialias=True), quantiles=q, rep=300)
    # speed of light: bytes / 1008 GB/s
    res["sol_pre_ms"] = (x.numel() * 1 + x.numel() * 2) / 1008e9 * 1e3
    res["sol_post_ms"] = (y.numel() * 4 + y.numel() * 1) / 1008e9 * 1e3
    res["speedup_pre"] = res["pre_torch_ms"][0] / res["pre_triton_ms"][0]
    res["speedup_post"] = res["post_torch_ms"][0] / res["post_triton_ms"][0]
    res["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    res["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    print(json.dumps(res, indent=1))
    return 0 if res["pre_pass"] and res["post_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

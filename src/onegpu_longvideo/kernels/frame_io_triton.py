"""Fused Triton kernels for the frame I/O of upscale.upscale().

Replaces two PyTorch op chains in the upscale stream:
  pre : uint8 NHWC -> permute -> half -> div_(255)            (3 kernels + 1 copy)
  post: clamp_(0,1) -> mul_(255) -> round_() -> byte() -> permute(0,2,3,1).contiguous()
        (5 kernels, 4 full-size float32 passes)
Each becomes one pass over memory. Bit-exact by design (same fp32 math, rint = round-half-even).
Optional: callers fall back to PyTorch when triton is missing.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _u8_nhwc_to_f16_nchw(x_ptr, o_ptr, npix, hw, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c = tl.arange(0, 4)
    m = (offs[:, None] < npix) & (c[None, :] < 3)
    v = tl.load(x_ptr + offs[:, None] * 3 + c[None, :], mask=m, other=0)
    v = (v.to(tl.float32).to(tl.float16) / 255.0).to(tl.float16)
    n = offs // hw
    p = offs % hw
    tl.store(o_ptr + (n * 3)[:, None] * hw + c[None, :] * hw + p[:, None], v, mask=m)


@triton.jit
def _f32_nchw_to_u8_nhwc(y_ptr, o_ptr, npix, hw, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c = tl.arange(0, 4)
    m = (offs[:, None] < npix) & (c[None, :] < 3)
    n = offs // hw
    p = offs % hw
    v = tl.load(y_ptr + (n * 3)[:, None] * hw + c[None, :] * hw + p[:, None], mask=m, other=0.0)
    v = tl.minimum(tl.maximum(v, 0.0), 1.0) * 255.0
    v = libdevice.rint(v)
    tl.store(o_ptr + offs[:, None] * 3 + c[None, :], v.to(tl.uint8), mask=m)


def u8_nhwc_to_f16_nchw(x: torch.Tensor, block: int = 1024) -> torch.Tensor:
    """x: cuda uint8 (N,H,W,3) contiguous -> float16 (N,3,H,W) in [0,1]."""
    assert x.is_cuda and x.dtype == torch.uint8 and x.is_contiguous() and x.shape[-1] == 3
    n, h, w, _ = x.shape
    o = torch.empty((n, 3, h, w), device=x.device, dtype=torch.float16)
    npix = n * h * w
    _u8_nhwc_to_f16_nchw[(triton.cdiv(npix, block),)](x, o, npix, h * w, BLOCK=block)
    return o


def f32_nchw_to_u8_nhwc(y: torch.Tensor, block: int = 1024) -> torch.Tensor:
    """y: cuda float32 (N,3,H,W) contiguous -> uint8 (N,H,W,3) = round(clamp(y,0,1)*255)."""
    assert y.is_cuda and y.dtype == torch.float32 and y.is_contiguous() and y.shape[1] == 3
    n, _, h, w = y.shape
    o = torch.empty((n, h, w, 3), device=y.device, dtype=torch.uint8)
    npix = n * h * w
    _f32_nchw_to_u8_nhwc[(triton.cdiv(npix, block),)](y, o, npix, h * w, BLOCK=block)
    return o

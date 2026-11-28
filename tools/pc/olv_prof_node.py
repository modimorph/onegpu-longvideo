"""ComfyUI custom node: torch.profiler trace of ONE sampler model call (measurement only; inert unless a graph uses it).

OLVProfileStep(model, step, out_dir) -> model. Wraps the model function; call number `step` (0-based) runs under
torch.profiler (CPU + CUDA), then writes <out_dir>/step<k>_kernels.txt (top kernels by self CUDA time),
<out_dir>/step<k>_summary.json (wall, CUDA busy, buckets: int8/int4 GEMM, attention, memcpy H2D, other) and a chrome trace.
Every call's wall time (synchronized) is also logged. Copy into ComfyUI/custom_nodes/ to use.
"""
import json, os, time

import torch


def _bucket(name):
    n = name.lower()
    if "memcpy" in n and ("htod" in n or "host" in n):
        return "memcpy_h2d"
    if "memcpy" in n or "memset" in n:
        return "memcpy_other"
    if any(k in n for k in ("attn", "attention", "flash", "fmha", "sparse", "vsa", "bsa", "block_sparse", "softmax")):
        return "attention"
    if any(k in n for k in ("int4", "s4", "w4a4", "convrot")):
        return "gemm_int4_or_rot"
    if any(k in n for k in ("gemm", "cutlass", "imma", "igemm", "int8", "s8", "sm80_xmma", "sm89", "ampere", "matmul", "cublas")):
        return "gemm"
    if any(k in n for k in ("quant", "hadamard", "rotate")):
        return "quant_rot"
    return "other"


class OLVProfileStep:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "step": ("INT", {"default": 3, "min": 0, "max": 64}),
                             "out_dir": ("STRING", {"default": "C:/cluster/olv-ab/out/trace"})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "onegpu/measure"

    def patch(self, model, step, out_dir):
        m = model.clone()
        prev = m.model_options.get("model_function_wrapper")
        state = {"n": 0, "walls": []}
        os.makedirs(out_dir, exist_ok=True)

        def call(apply_model, args):
            if prev is not None:
                return prev(apply_model, args)
            return apply_model(args["input"], args["timestep"], **args["c"])

        def wrapper(apply_model, args):
            k = state["n"]; state["n"] += 1
            torch.cuda.synchronize(); t0 = time.perf_counter()
            if k != step:
                out = call(apply_model, args)
                torch.cuda.synchronize(); state["walls"].append(round(time.perf_counter() - t0, 3))
                json.dump({"walls": state["walls"]}, open(os.path.join(out_dir, "walls.json"), "w"))
                return out
            acts = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            with torch.profiler.profile(activities=acts, record_shapes=False, with_stack=False) as prof:
                out = call(apply_model, args)
                torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            state["walls"].append(round(wall, 3))
            ev = [e for e in prof.key_averages() if getattr(e, "self_device_time_total", 0) > 0]
            ev.sort(key=lambda e: e.self_device_time_total, reverse=True)
            buckets, total = {}, 0.0
            for e in ev:
                us = e.self_device_time_total; total += us
                b = _bucket(e.key); buckets[b] = buckets.get(b, 0.0) + us
            with open(os.path.join(out_dir, f"step{k}_kernels.txt"), "w", encoding="utf-8") as f:
                f.write(f"wall {wall:.3f} s, CUDA self total {total / 1e6:.3f} s, kernels {len(ev)}\n")
                for e in ev[:80]:
                    f.write(f"{e.self_device_time_total / 1e3:10.1f} ms  {e.count:6d}x  [{_bucket(e.key)}]  {e.key[:160]}\n")
            info = {"step": k, "wall_s": round(wall, 3), "cuda_busy_s": round(total / 1e6, 3),
                    "idle_s": round(wall - total / 1e6, 3),
                    "buckets_s": {b: round(v / 1e6, 3) for b, v in sorted(buckets.items(), key=lambda x: -x[1])},
                    "x_shape": list(args["input"].shape) if hasattr(args["input"], "shape") else None,
                    "max_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                    "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2)}
            json.dump(info, open(os.path.join(out_dir, f"step{k}_summary.json"), "w"), indent=1)
            try:
                prof.export_chrome_trace(os.path.join(out_dir, f"step{k}_trace.json"))
            except Exception:
                pass
            return out

        m.set_model_unet_function_wrapper(wrapper)
        return (m,)


class OLVSink:
    """Output node that accepts anything and does nothing (times a sub-graph such as the conditioning alone)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("*",)}}

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "onegpu/measure"

    def run(self, value):
        torch.cuda.synchronize()
        return {}


NODE_CLASS_MAPPINGS = {"OLVProfileStep": OLVProfileStep, "OLVSink": OLVSink}
NODE_DISPLAY_NAME_MAPPINGS = {"OLVProfileStep": "OLV profile one sampler step", "OLVSink": "OLV sink (measurement)"}

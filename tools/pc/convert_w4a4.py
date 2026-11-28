"""Convert a ComfyUI int8_tensorwise+convrot DiT checkpoint to convrot_w4a4 (INT4 weights, INT4 activations, int4 tensor-core GEMM).

Runs on the GPU box inside the ComfyUI venv (needs comfy_kitchen). Streams one module at a time; non-quantized tensors and the
safetensors metadata are copied unchanged. Source weights are dequantized from int8 (rotation undone) and re-quantized with
comfy_kitchen's own ConvRot W4A4 quantizer, so the file follows ComfyUI's QUANTIZATION.md layout.

usage: python convert_w4a4.py SRC.safetensors DST.safetensors [--linear-dtype int4|int8] [--check N] [--bench]
  --check N  report the relative weight error (W4A4 vs int8 dequant) on every N-th module
  --bench    time one real-shape linear (int8 convrot vs W4A4) and compare outputs
"""
import argparse, json, time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import comfy_kitchen  # noqa: F401  (registers torch.ops.comfy_kitchen)
from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout
from comfy_kitchen.tensor.convrot_w4a4 import TensorCoreConvRotW4A4Layout


def cfg_of(f, k):
    return json.loads(bytes(f.get_tensor(k).tolist()).decode())


def int8_to_float(q, s, gs, dev):
    p = TensorWiseINT8Layout.Params(scale=s.to(dev), orig_dtype=torch.bfloat16, orig_shape=tuple(q.shape), convrot=True, convrot_groupsize=gs)
    return TensorWiseINT8Layout.dequantize(q.to(dev), p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--linear-dtype", default="int4", choices=["int4", "int8"])
    ap.add_argument("--check", type=int, default=16)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--device", default="cuda", help="cuda, or cpu (kitchen eager kernels) to convert while the GPU is busy")
    ap.add_argument("--limit", type=int, default=0, help="convert only the first N modules (smoke test; no file written)")
    a = ap.parse_args()
    import os
    if os.path.abspath(a.src) == os.path.abspath(a.dst) or (os.path.exists(a.dst) and os.path.samefile(a.src, a.dst)):
        raise SystemExit("DST must differ from SRC (also via links)")
    dev = a.device
    out, errs, t0 = {}, [], time.perf_counter()
    with safe_open(a.src, "pt", device="cpu") as f:
        meta = f.metadata() or {}
        keys = list(f.keys())
        qmods = [k[: -len(".comfy_quant")] for k in keys if k.endswith(".comfy_quant")]
        qset = set(qmods)
        legacy = json.loads(meta.get("_quantization_metadata") or "{}").get("layers") or {}
        if not qmods or legacy:
            raise SystemExit("source has no per-module comfy_quant entries or carries a non-empty legacy layer map: not supported")
        for k in keys:  # copy everything that is not a quantized module's weight / scale / config
            m = k.rsplit(".", 1)[0]
            if m in qset and k.split(".")[-1] in ("weight", "weight_scale", "comfy_quant"):
                continue
            out[k] = f.get_tensor(k)
        for i, m in enumerate(qmods[: a.limit] if a.limit else qmods):
            c = cfg_of(f, m + ".comfy_quant")
            assert c.get("format") == "int8_tensorwise" and c.get("convrot"), (m, c)
            gs = int(c.get("convrot_groupsize", 256))
            w = int8_to_float(f.get_tensor(m + ".weight"), f.get_tensor(m + ".weight_scale"), gs, dev)
            assert w.shape[1] % 64 == 0, (m, w.shape)
            qd, prm = TensorCoreConvRotW4A4Layout.quantize(w, convrot_groupsize=gs, linear_dtype=a.linear_dtype)
            if a.check and i % a.check == 0:
                back = TensorCoreConvRotW4A4Layout.dequantize(qd, prm).float()
                errs.append((m, round(((back - w.float()).norm() / w.float().norm()).item(), 5)))
            out[m + ".weight"] = qd.cpu().contiguous()
            out[m + ".weight_scale"] = prm.scale.cpu().contiguous()
            nc = {"format": "convrot_w4a4", "convrot_groupsize": gs}
            if a.linear_dtype != "int4":
                nc["linear_dtype"] = a.linear_dtype
            out[m + ".comfy_quant"] = torch.tensor(list(json.dumps(nc).encode()), dtype=torch.uint8)
            del w, qd, prm
    meta = {k: v for k, v in meta.items() if k != "_quantization_metadata"}  # legacy map would override the new per-module configs
    meta = dict(meta, w4a4_source=a.src.replace("\\", "/").rsplit("/", 1)[-1], w4a4_tool="onegpu-longvideo tools/pc/convert_w4a4.py")
    if a.limit:
        print(json.dumps({"smoke_modules": a.limit, "seconds": round(time.perf_counter() - t0, 1), "rel_err_samples": errs})); return
    save_file(out, a.dst, metadata=meta)  # save first: the bench must not be able to lose the conversion
    del out
    torch.cuda.empty_cache()
    bench = None
    if a.bench:
        with safe_open(a.src, "pt", device="cpu") as f:
            bench = bench_one(f, qmods, "cuda", linear_dtype=a.linear_dtype)
    print(json.dumps({"modules": len(qmods), "seconds": round(time.perf_counter() - t0, 1), "rel_err_samples": errs,
                      "rel_err_max": max((e for _, e in errs), default=None), "bench": bench}))


def bench_one(f, qmods, dev, M=16384, linear_dtype="int4"):
    """One real-shape linear: int8 convrot (reference) vs W4A4, CUDA events, median of 20 after 3 warm-ups."""
    shp = {x: f.get_slice(x + ".weight").get_shape() for x in qmods}
    m = max(qmods, key=lambda x: shp[x][0] * shp[x][1])  # largest linear
    q, s = f.get_tensor(m + ".weight"), f.get_tensor(m + ".weight_scale")
    gs = int(cfg_of(f, m + ".comfy_quant").get("convrot_groupsize", 256))
    p8 = TensorWiseINT8Layout.Params(scale=s.to(dev), orig_dtype=torch.bfloat16, orig_shape=tuple(q.shape), convrot=True, convrot_groupsize=gs)
    w8 = QuantizedTensor(q.to(dev), "TensorWiseINT8Layout", p8)
    w = TensorWiseINT8Layout.dequantize(q.to(dev), p8)
    qd, p4 = TensorCoreConvRotW4A4Layout.quantize(w, convrot_groupsize=gs, linear_dtype=linear_dtype)
    w4 = QuantizedTensor(qd, "TensorCoreConvRotW4A4Layout", p4)
    g = torch.Generator(device=dev).manual_seed(7329)
    x = torch.randn(M, q.shape[1], device=dev, dtype=torch.bfloat16, generator=g)
    ref = torch.nn.functional.linear(x, w)  # bf16 dense reference

    def t(fn):
        for _ in range(3):
            fn()
        ts = []
        for _ in range(20):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record(); fn(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
        ts.sort(); return [round((ts[9] + ts[10]) / 2, 3), round(ts[4], 3), round(ts[15], 3)]  # median, p20, p80 of 20

    y8 = torch.nn.functional.linear(x, w8); y4 = torch.nn.functional.linear(x, w4)
    def rel(y):  # chunked so no full fp32 copy of the [M, N] output is live
        num = sum(((y[i:i + 2048].float() - ref[i:i + 2048].float()) ** 2).sum().item() for i in range(0, M, 2048))
        den = sum((ref[i:i + 2048].float() ** 2).sum().item() for i in range(0, M, 2048))
        return round((num / den) ** 0.5, 5)
    return {"module": m, "shape": [M, q.shape[1], q.shape[0]], "ms_bf16": t(lambda: torch.nn.functional.linear(x, w)),
            "ms_int8": t(lambda: torch.nn.functional.linear(x, w8)), "ms_w4a4": t(lambda: torch.nn.functional.linear(x, w4)),
            "rel_err_int8_vs_bf16": rel(y8), "rel_err_w4a4_vs_bf16": rel(y4)}


if __name__ == "__main__":
    main()

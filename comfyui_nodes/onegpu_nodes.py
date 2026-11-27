"""onegpu-longvideo custom ComfyUI nodes (MODEL -> MODEL patches for the MiniMax-H3 / FastH3 DiT in ComfyUI).

Install: copy this file into ComfyUI/custom_nodes/ and restart ComfyUI. The file is inert unless a graph uses one of its nodes.
Written against ComfyUI 0.41.0 (comfy/ldm/minimax/model.py). Other versions may need changes.

Nodes:
  FLSageVideoRows  SageAttention2 on the VIDEO query rows only; text, reference and target-audio query rows stay dense (SDPA)
                   against the full K/V. Explicit kernel, smooth_k and PV precision. Optional dense blocks and a sigma window.
  FLChunkFFN       SwiGLU MLP over the token axis in chunks (cuts the peak memory of the wide fc1 output by the chunk count).
                   Not bit-identical when activations use per-tensor int8 scales: gate it like any numeric change.
  FLTemporalRoPE   Temporal-axis RoPE change (RIFLEx-style): one t component (default k = 8, period 15.7 s at 40 rotary units
                   per second) gets a longer period; h and w columns untouched. In our tests it made speech WORSE (see docs/EXPERIMENTS.md).
"""
import math, logging

try:
    import torch
    import torch.nn.functional as F
except Exception:  # lets the offline test import the pure-math helper without torch
    torch = None


def parse_blocks(text):
    out = set()
    for part in (text or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = (int(x) for x in part.split("-"))
            out.update(range(min(a, b), max(a, b) + 1))
        else:
            out.add(int(part))
    return out


def temporal_inv(inv, mode, k, value, frames):
    """Return a copy of the 16 inverse frequencies for the t axis. Works on lists (offline test) and tensors."""
    inv_t = [float(x) for x in inv] if not hasattr(inv, "clone") else inv.clone()
    if mode == "off":
        return inv_t
    if mode == "riflex":        # RIFLEx: theta_k = 0.9 * 2 pi / L, L = test length in rotary units (5/3 per pixel frame)
        if frames <= 0:
            raise ValueError("frames must be > 0")
        L = frames * 5.0 / 3.0
        inv_t[k] = 0.9 * 2 * math.pi / L
    elif mode == "yarn":        # YaRN / NTK-by-parts on the t axis (27 Nov 2026, GIBBERISH-FIX H6): components whose period exceeds the
        # trained length (value s, 40 units/s) never saw those phase gaps; interpolate them by L_train/L_test, keep the fast ones.
        # k = beta: 1 = hard cut (only period > L_train scaled); >1 = linear ramp over r = L_train/period in [1, beta].
        L_tr = (value if value > 0 else 15.0) * 40.0
        L_te = frames * 5.0 / 3.0
        s = L_te / L_tr
        if s > 1.0:
            beta = max(1, int(k))
            for i in range(len(inv_t)):
                r = L_tr * float(inv_t[i]) / (2 * math.pi)
                g = (1.0 if r >= 1.0 else 0.0) if beta == 1 else min(1.0, max(0.0, (r - 1.0) / (beta - 1.0)))
                inv_t[i] = float(inv_t[i]) * ((1.0 - g) / s + g)
    elif mode == "set_period":  # causal test: put component k at a chosen period in seconds (40 units per second)
        if value <= 0:
            raise ValueError("period must be > 0 s")
        inv_t[k] = 2 * math.pi / (value * 40.0)
    else:
        raise ValueError(mode)
    return inv_t


class FLSageVideoRows:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "pv": (["fp16acc32", "fp16acc16", "fp8acc32", "fp8acc16"], {"default": "fp16acc32"}),
            "smooth_k": ("BOOLEAN", {"default": True}),
            "dense_blocks": ("STRING", {"default": ""}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "fastlong"

    def patch(self, model, pv, smooth_k, dense_blocks, start_percent, end_percent):
        from sageattention import sageattn_qk_int8_pv_fp16_cuda, sageattn_qk_int8_pv_fp8_cuda
        kern, acc = {"fp16acc32": (sageattn_qk_int8_pv_fp16_cuda, "fp32"), "fp16acc16": (sageattn_qk_int8_pv_fp16_cuda, "fp16"),
                     "fp8acc32": (sageattn_qk_int8_pv_fp8_cuda, "fp32+fp32"), "fp8acc16": (sageattn_qk_int8_pv_fp8_cuda, "fp32+fp16")}[pv]
        ms = model.get_model_object("model_sampling")
        s_hi, s_lo = float(ms.percent_to_sigma(start_percent)), float(ms.percent_to_sigma(end_percent))
        blocks = parse_blocks(dense_blocks)
        m = model.clone()
        to_root = m.model_options.setdefault("transformer_options", {})
        prev = to_root.get("optimized_attention_override")
        logged = set()

        def log_once(key, msg):
            if key not in logged:
                logged.add(key); logging.info("[FLSageVideoRows] " + msg)

        def override(func, q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kw):
            to = kw.get("transformer_options") or {}

            def dense():
                a = dict(mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kw)
                return prev(func, q, k, v, heads, **a) if prev is not None else func(q, k, v, heads, **a)

            lay = to.get("minimax_h3_layout")
            if mask is not None or not skip_reshape or lay is None or q.shape[2] != lay.seq_len or q.shape[-1] != 128:
                log_once("skip", "dense: not an H3 self-attention call"); return dense()
            if to.get("block_index") in blocks:
                return dense()
            sig = to.get("sigmas")
            if sig is not None:
                s = float(sig.flatten()[0])
                if not (s_lo - 1e-6 <= s <= s_hi + 1e-6):
                    return dense()
            n0 = next(a for a, b, kind in lay.segments if kind == "video")
            scale = kw.get("scale")
            ov = kern(q[:, :, n0:].contiguous(), k, v, tensor_layout="HND", is_causal=False, sm_scale=scale,
                      pv_accum_dtype=acc, smooth_k=bool(smooth_k))
            if n0:
                op = F.scaled_dot_product_attention(q[:, :, :n0], k, v, scale=scale)
                out = torch.cat((op, ov.to(op.dtype)), dim=2)
            else:
                out = ov
            log_once("on", f"sage {pv} smooth_k={smooth_k} on {q.shape[2] - n0} video rows, {n0} prefix rows dense")
            if skip_output_reshape:
                return out
            return out.transpose(1, 2).reshape(q.shape[0], -1, heads * q.shape[-1])

        to_root["optimized_attention_override"] = override
        return (m,)


class FLChunkFFN:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "chunk_tokens": ("INT", {"default": 16384, "min": 1024, "max": 1 << 20, "step": 1024})}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "fastlong"

    def patch(self, model, chunk_tokens):
        import comfy.ops
        m = model.clone()
        dm = m.get_model_object("diffusion_model")
        for i in range(len(dm.blocks)):
            mlp = m.get_model_object(f"diffusion_model.blocks.{i}.mlp")

            def fwd(x, _mlp=mlp, _n=int(chunk_tokens)):
                if x.shape[0] <= _n:
                    return comfy.ops.linear_input_act(_mlp.fc2, _mlp.fc1(x), "swiglu")
                out = None
                for j in range(0, x.shape[0], _n):
                    y = comfy.ops.linear_input_act(_mlp.fc2, _mlp.fc1(x[j:j + _n]), "swiglu")
                    if out is None:   # preallocate once (no list + cat copy of the full output)
                        out = y.new_empty((x.shape[0],) + tuple(y.shape[1:]))
                    out[j:j + y.shape[0]] = y
                return out
            m.add_object_patch(f"diffusion_model.blocks.{i}.mlp.forward", fwd)
        logging.info(f"[FLChunkFFN] {len(dm.blocks)} MLPs chunked at {chunk_tokens} tokens")
        return (m,)


class FLTemporalRoPE:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",),
                             "mode": (["off", "riflex", "set_period", "yarn"], {"default": "riflex"}),
                             "k": ("INT", {"default": 8, "min": 0, "max": 15}),
                             "value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                             "frames": ("INT", {"default": 719, "min": 5, "max": 4000})}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "fastlong"

    def patch(self, model, mode, k, value, frames):
        import comfy.model_management as mm
        m = model.clone()
        dm = m.get_model_object("diffusion_model")

        def rope_freqs(position_ids, device, _dm=dm):
            pos = position_ids.to(torch.float32).to(device)
            inv = mm.cast_to(_dm.rope.inv_freq, device=device).to(torch.float32)
            inv_t = temporal_inv(inv, mode, k, value, frames)
            half = torch.cat((pos[:, 0:1] * inv_t.view(1, -1), pos[:, 1:2] * inv.view(1, -1), pos[:, 2:3] * inv.view(1, -1)), dim=-1)
            return torch.cat((half, half), dim=-1)
        m.add_object_patch("diffusion_model.rope_freqs", rope_freqs)
        logging.info(f"[FLTemporalRoPE] mode={mode} k={k} value={value} frames={frames}")
        return (m,)


NODE_CLASS_MAPPINGS = {"FLSageVideoRows": FLSageVideoRows, "FLChunkFFN": FLChunkFFN, "FLTemporalRoPE": FLTemporalRoPE}
NODE_DISPLAY_NAME_MAPPINGS = {"FLSageVideoRows": "FL Sage on video rows", "FLChunkFFN": "FL chunked FFN", "FLTemporalRoPE": "FL temporal RoPE (RIFLEx-T)"}

"""ComfyUI API-graph builder for one-shot audio-video generation (the "card graph").

Chain: UNET -> [identity LoRA] -> [Turbo LoRA] -> [realism LoRA] -> sigma shift -> [chunked FFN] -> [attention patch] -> guider.
Every node except FLChunkFFN / FLSageVideoRows (comfyui_nodes/onegpu_nodes.py) ships with ComfyUI 0.41
(BlockSparseAttention is ComfyUI's sparse-attention node; check your version has it).
"""
from __future__ import annotations

import copy

from .frames import align_frames
from .prompt import REALISM_TRIGGER
from .recipe import Recipe

NATIVE = {
    "UNETLoader", "CLIPLoader", "CLIPLoaderGGUF", "VAELoader", "RandomNoise", "KSamplerSelect", "BasicScheduler",
    "MiniMaxH3ReferenceToVideo", "BasicGuider", "CFGGuider", "SamplerCustomAdvanced", "VAEDecode", "VAEDecodeTiled",
    "VAEDecodeAudio", "CreateVideo", "SaveVideo", "LoraLoaderModelOnly", "MiniMaxH3SigmaShift", "BlockSparseAttention",
    "LoadImage", "LoadAudio", "LoadVideo", "GetVideoComponents", "MiniMaxH3AddGuide", "TorchCompileModel",
}
OURS = {"FLChunkFFN", "FLSageVideoRows", "FLTemporalRoPE", "OLVUpscaleVideoSave"}


def _model_patch(g: dict, tip: list, nid: str, node: dict) -> list:
    node["inputs"]["model"] = tip
    g[nid] = node
    return [nid, 0]


def build(prompt: str, r: Recipe, output_prefix: str = "onegpu/render") -> dict:
    """Return an API-format graph (dict of node id -> {class_type, inputs})."""
    m = r.models
    if r.realism and REALISM_TRIGGER not in prompt:
        prompt = f"{REALISM_TRIGGER} {prompt}"
    g: dict = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": m.unet, "weight_dtype": "default"}},
        "clip": {"class_type": m.text_encoder_loader, "inputs": {"clip_name": m.text_encoder, "type": "minimax"}},
        "vvae": {"class_type": "VAELoader", "inputs": {"vae_name": m.video_vae}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": m.audio_vae}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(r.seed)}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": r.sampler}},
        "cond": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["clip", 0], "vae": ["vvae", 0], "audio_vae": ["avae", 0], "prompt": prompt,
            "width": int(r.width), "height": int(r.height), "length": align_frames(r.frames), "ref_image_size": "match"}},
    }
    tip = ["unet", 0]
    if m.identity_lora and r.identity_strength:
        tip = _model_patch(g, tip, "lora_id", {"class_type": "LoraLoaderModelOnly", "inputs": {
            "lora_name": m.identity_lora, "strength_model": float(r.identity_strength)}})
    if m.turbo_lora and r.turbo:
        tip = _model_patch(g, tip, "lora_turbo", {"class_type": "LoraLoaderModelOnly", "inputs": {
            "lora_name": m.turbo_lora, "strength_model": float(r.turbo)}})
    if r.realism:
        tip = _model_patch(g, tip, "lora_real", {"class_type": "LoraLoaderModelOnly", "inputs": {
            "lora_name": m.realism_lora, "strength_model": float(r.realism)}})
    tip = _model_patch(g, tip, "shift", {"class_type": "MiniMaxH3SigmaShift", "inputs": {
        "shift_video": float(r.shift_video), "shift_audio": float(r.shift_audio)}})
    sched_model = tip
    if r.ffn_chunk_tokens:
        tip = _model_patch(g, tip, "fl_ffn", {"class_type": "FLChunkFFN", "inputs": {"chunk_tokens": int(r.ffn_chunk_tokens)}})
    if r.attention == "vsa":
        tip = _model_patch(g, tip, "bsa", {"class_type": "BlockSparseAttention", "inputs": {
            "start_percent": 0.0, "end_percent": 1.0, "min_tokens": 12288, "extra_tokens": 0, "dense_blocks": "",
            "sink_conditioning": "exact_kv_and_rows", "verbose": False, "selection": "vsa",
            "selection.keep_percent": float(r.vsa_keep_percent)}})
    elif r.attention == "sage":
        tip = _model_patch(g, tip, "fl_sage", {"class_type": "FLSageVideoRows", "inputs": {
            "pv": "fp16acc32", "smooth_k": True, "dense_blocks": "", "start_percent": 0.0, "end_percent": 1.0}})
    g["sigmas"] = {"class_type": "BasicScheduler", "inputs": {"scheduler": "simple", "steps": int(r.steps), "denoise": 1.0, "model": sched_model}}
    if abs(r.guidance - 1.0) < 1e-9:
        g["guider"] = {"class_type": "BasicGuider", "inputs": {"model": tip, "conditioning": ["cond", 0]}}
    else:
        g["guider"] = {"class_type": "CFGGuider", "inputs": {"model": tip, "positive": ["cond", 0], "negative": ["cond", 0], "cfg": float(r.guidance)}}
    for i, name in enumerate(r.image_refs):
        g[f"img{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        g["cond"]["inputs"][f"ref_images.ref_image_{i}"] = [f"img{i}", 0]
    if r.voice_ref:
        g["aud0"] = {"class_type": "LoadAudio", "inputs": {"audio": r.voice_ref}}
        g["cond"]["inputs"]["ref_audios.ref_audio_0"] = ["aud0", 0]
    g["sample"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                                                                       "sigmas": ["sigmas", 0], "latent_image": ["cond", 1]}}
    if r.tiled_decode:
        g["vdec"] = {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["sample", 0], "vae": ["vvae", 0], "tile_size": 512, "overlap": 64,
                                                                 "temporal_size": 64, "temporal_overlap": 8}}
    else:
        g["vdec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vvae", 0]}}
    g["adec"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sample", 0], "vae": ["avae", 0]}}
    if r.upscale_in_graph:  # x2 upscale + encode inside the graph (comfyui_nodes/onegpu_upscale.py): no raw-mp4 round trip
        w, h = (int(v) for v in r.upscale_to)
        g["save"] = {"class_type": "OLVUpscaleVideoSave", "inputs": {"images": ["vdec", 0], "audio": ["adec", 0], "upscale_model": r.upscale_model,
                                                                      "width": w, "height": h, "fps": 24.0, "quality": 18,
                                                                      "filename_prefix": output_prefix + "_1080p", "save_raw": True}}
        return g
    g["video"] = {"class_type": "CreateVideo", "inputs": {"fps": 24, "images": ["vdec", 0], "audio": ["adec", 0]}}
    g["save"] = {"class_type": "SaveVideo", "inputs": {"filename_prefix": output_prefix, "format": "auto", "codec": "auto", "video": ["video", 0]}}
    return g


def add_video_guide(g: dict, tail_video: str, frame_idx: int = 0) -> dict:
    """Chain mode: condition on the tail of the previous segment (frames + audio) with MiniMaxH3AddGuide."""
    g = copy.deepcopy(g)
    g["tailv"] = {"class_type": "LoadVideo", "inputs": {"file": tail_video}}
    g["tailc"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["tailv", 0]}}
    g["guide"] = {"class_type": "MiniMaxH3AddGuide", "inputs": {"positive": ["cond", 0], "vae": ["vvae", 0], "audio_vae": ["avae", 0],
                                                                "latent": ["cond", 1], "image": ["tailc", 0], "audio": ["tailc", 1], "frame_idx": frame_idx}}
    gi = g["guider"]["inputs"]
    gi["conditioning" if "conditioning" in gi else "positive"] = ["guide", 0]
    return g


def validate(g: dict) -> list[str]:
    """Static checks: known node classes and no dangling links."""
    errs = []
    for nid, n in g.items():
        if n.get("class_type") not in NATIVE | OURS:
            errs.append(f"{nid}: unknown class {n.get('class_type')}")
        for k, v in n.get("inputs", {}).items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], int) and v[0] not in g:
                errs.append(f"{nid}.{k}: link to missing node {v[0]}")
    return errs


def uses_custom_nodes(g: dict) -> bool:
    return any(n["class_type"] in OURS for n in g.values())

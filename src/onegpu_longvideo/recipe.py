"""Recipe = every knob of one render, loaded from YAML. Defaults are the settings of our published samples."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from .frames import frames_for_seconds


@dataclass
class Models:
    # W4A4 (INT4 weights + activations, int4 tensor cores) made from the int8 file by tools/pc/convert_w4a4.py.
    # int8 original: "fasth3_omniref_8step_pruned_int8_convrot.safetensors".
    unet: str = "fasth3_omniref_8step_pruned_w4a4_convrot.safetensors"
    text_encoder: str = "qwen3vl_32b_minimax_h3-Q4_K_M.gguf"
    text_encoder_loader: str = "CLIPLoaderGGUF"  # or CLIPLoader for safetensors
    # int8 decoder (ComfyUI >= 0.36 fused int8 kernels: int8 GEMM with folded norm/act/skip, int8 attention).
    # Loads on stock ComfyUI >= 0.36; configs/comfyui_flags.txt is a separate, optional speed flag. fp16 file = "minimax_h3_video_vae_fp16.safetensors".
    video_vae: str = "minimax_h3_video_vae_int8_convrot.safetensors"
    audio_vae: str = "minimax_h3_audio_vae_fp32.safetensors"
    realism_lora: str = "h3-realism-people-t2v-i2v-r2v.safetensors"
    turbo_lora: str = ""
    identity_lora: str = ""  # your own single-identity LoRA trained on consented data; empty = none


@dataclass
class Recipe:
    name: str = "render"
    seconds: float = 29.96
    width: int = 1280
    height: int = 720
    frames: int = 0  # 0 = derived from seconds
    seed: int = 7329
    steps: int = 8
    sampler: str = "euler"
    guidance: float = 1.0  # 1.0 = BasicGuider (no unconditional pass)
    shift_video: float = 12.0
    shift_audio: float = 3.0
    realism: float = 0.3
    turbo: float = 0.0
    identity_strength: float = 0.0
    attention: str = "vsa"  # vsa | dense | sage
    vsa_keep_percent: float = 10.0
    ffn_chunk_tokens: int = 16384  # 0 = off (needs comfyui_nodes/onegpu_nodes.py)
    tiled_decode: bool = True
    voice_ref: str = ""  # file name already uploaded to ComfyUI input/
    image_refs: list[str] = field(default_factory=list)
    upscale_to: list[int] = field(default_factory=lambda: [1920, 1080])
    upscale_model: str = "2xNomosUni_span_multijpg.safetensors"
    prompt_file: str = ""
    models: Models = field(default_factory=Models)

    def __post_init__(self):
        if isinstance(self.models, dict):
            self.models = Models(**self.models)
        if not self.frames:
            self.frames = frames_for_seconds(self.seconds)
        if self.width % 32 or self.height % 16:
            raise ValueError(f"size {self.width}x{self.height}: width must be a multiple of 32, height of 16")
        if self.attention not in ("vsa", "dense", "sage"):
            raise ValueError(f"attention must be vsa, dense or sage, not {self.attention!r}")

    def to_dict(self) -> dict:
        return asdict(self)


def load(path: str | Path) -> Recipe:
    import yaml

    data = yaml.safe_load(Path(path).read_text()) or {}
    return Recipe(**data)

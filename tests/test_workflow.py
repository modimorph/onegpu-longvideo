from pathlib import Path

import pytest

from onegpu_longvideo import recipe as R
from onegpu_longvideo import workflow as W

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_default_graph_valid():
    g = W.build("a potter at a wheel", R.Recipe())
    assert W.validate(g) == []
    assert g["cond"]["inputs"]["length"] == 719
    assert g["cond"]["inputs"]["prompt"].startswith("r34l1sm ")
    assert g["guider"]["class_type"] == "BasicGuider"
    assert g["guider"]["inputs"]["model"] == ["bsa", 0]
    assert g["bsa"]["inputs"]["model"] == ["fl_ffn", 0]
    assert g["sigmas"]["inputs"]["model"] == ["shift", 0]
    assert W.uses_custom_nodes(g)


def test_dense_no_custom_nodes_no_realism():
    g = W.build("x", R.Recipe(attention="dense", ffn_chunk_tokens=0, realism=0, upscale_in_graph=False))
    assert W.validate(g) == [] and not W.uses_custom_nodes(g)
    assert g["guider"]["inputs"]["model"] == ["shift", 0]
    assert "lora_real" not in g and not g["cond"]["inputs"]["prompt"].startswith("r34l1sm")


def test_cfg_guider_and_refs():
    g = W.build("x", R.Recipe(guidance=0.9, voice_ref="v.wav", image_refs=["a.png", "b.png"]))
    assert g["guider"]["class_type"] == "CFGGuider" and g["guider"]["inputs"]["cfg"] == 0.9
    assert g["cond"]["inputs"]["ref_audios.ref_audio_0"] == ["aud0", 0]
    assert g["cond"]["inputs"]["ref_images.ref_image_1"] == ["img1", 0]
    assert W.validate(g) == []


def test_lora_chain_order():
    m = R.Models(identity_lora="my_identity.safetensors", turbo_lora="turbo.safetensors")
    g = W.build("x", R.Recipe(models=m, identity_strength=0.8, turbo=0.5))
    assert g["lora_turbo"]["inputs"]["model"] == ["lora_id", 0]
    assert g["lora_real"]["inputs"]["model"] == ["lora_turbo", 0]
    assert g["shift"]["inputs"]["model"] == ["lora_real", 0]


def test_video_guide():
    g = W.add_video_guide(W.build("x", R.Recipe()), "tail.mp4")
    assert g["guider"]["inputs"]["conditioning"] == ["guide", 0] and W.validate(g) == []


def test_validate_catches_dangling():
    g = W.build("x", R.Recipe())
    g["save"]["inputs"]["video"] = ["nope", 0]
    assert W.validate(g)


def test_recipe_rejects_bad_size():
    with pytest.raises(ValueError):
        R.Recipe(width=1000)


@pytest.mark.parametrize("cfg", sorted(p.name for p in CONFIGS.glob("*.yaml") if "voice_post" not in p.name))
def test_shipped_recipes_load_and_build(cfg):
    rc = R.load(CONFIGS / cfg)
    pr = (CONFIGS / rc.prompt_file).read_text()
    assert W.validate(W.build(pr, rc)) == []


def test_upscale_in_graph_replaces_save():
    g = W.build("x", R.Recipe())
    assert g["save"]["class_type"] == "OLVUpscaleVideoSave" and "video" not in g
    assert g["save"]["inputs"]["images"] == ["vdec", 0] and g["save"]["inputs"]["audio"] == ["adec", 0]
    assert (g["save"]["inputs"]["width"], g["save"]["inputs"]["height"]) == (1920, 1080)
    assert W.validate(g) == [] and W.uses_custom_nodes(g)
    g2 = W.build("x", R.Recipe(upscale_in_graph=False))
    assert g2["save"]["class_type"] == "SaveVideo"


def test_upscale_in_graph_size_and_name():
    import pytest as _pt
    g = W.build("x", R.Recipe(upscale_to=[1280, 720]))
    assert g["save"]["inputs"]["filename_prefix"].endswith("_720p")
    with _pt.raises(ValueError):
        W.build("x", R.Recipe(upscale_to=[32, 32]))

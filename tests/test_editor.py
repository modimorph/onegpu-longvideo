import json
import shutil
import subprocess

import pytest

from onegpu_longvideo import cli, profiles
from onegpu_longvideo import editor as ED

TH = profiles.load("talking_head")
HAS_FF = shutil.which("ffmpeg") and shutil.which("ffprobe")


def _w(words, t=0.3, d=0.3, gap=0.1):
    out = []
    for x in words.split():
        out.append({"w": x, "t0": round(t, 3), "t1": round(t + d, 3)})
        t += d + gap
    return out


def test_flaws_found_only_when_not_in_reference():
    ref = "we made one long shot on one card today"
    heard = _w("we we made one long shot on one long shot on one card um today")
    kinds = sorted(f["kind"] for f in ED.find_flaws(heard, ref))
    assert kinds == ["filler", "repeat", "stutter"]
    assert ED.find_flaws(_w(ref), ref) == []
    # a doubled word the script really has is not a stutter
    assert ED.find_flaws(_w("that that works"), "that that works") == []


def test_repeat_of_previous_clip_tail_is_cut():
    prev = [x["w"] for x in _w("and that is how it works")]
    f = ED.find_flaws(_w("that is how it works and next we test"), "and next we test", earlier=prev)
    assert f and f[0]["kind"] == "repeat" and f[0]["i0"] == 0


def test_cut_points_sit_between_words():
    heard = _w("we made made one long shot")
    cl = ED.cut_list(heard, 10.0, ED.find_flaws(heard, "we made one long shot"), TH)
    assert cl["in"] == pytest.approx(0.15) and cl["out"] == pytest.approx(heard[-1]["t1"] + 0.25)
    assert len(cl["segments"]) == 2 and not ED.cuts_in_silence(heard, cl["segments"])


def test_word_check_flags_gibberish(tmp_path):
    w = tmp_path / "wer.json"
    w.write_text(json.dumps({"wer": 0.2, "win_wer": [0, 0.4], "windows": [
        {"window": [0, 5], "wer": 0, "invented": 0}, {"window": [5, 10], "wer": 0.4, "invented": 4}]}))
    r = ED.word_check([{"clip": 1, "video": "x", "wer": str(w)}], TH)
    assert r["errors"][0]["type"] == "gibberish"
    assert ED.word_check([{"clip": 2, "video": "x", "wer": None}], TH)["errors"][0]["type"] == "no_wer_file"


def _clip(path, colour, seed):
    """3 s clip; pink noise (not a sine) so the A/V cross-correlation has one clear peak."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={colour}:s=320x180:r=24:d=3",
                    "-f", "lavfi", "-i", f"anoisesrc=d=3:c=pink:a=0.3:r=48000:seed={seed}", "-shortest",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)], check=True)


@pytest.mark.skipif(not HAS_FF, reason="needs ffmpeg")
def test_edit_end_to_end(tmp_path):
    clips = tmp_path / "clips"
    for i, (c, f) in enumerate([("0x806050", 440), ("0x5a4030", 550), ("0x907060", 660)], 1):
        _clip(clips / f"clip{i}" / "video.mp4", c, f)
        heard = _w(" ".join(f"w{i}x{k}" for k in range(6)), t=0.3, d=0.3, gap=0.05)
        (clips / f"clip{i}" / "wer.json").write_text(json.dumps({
            "wer": 0, "win_wer": [0], "windows": [{"window": [0, 3], "wer": 0, "invented": 0}], "last_word_s": heard[-1]["t1"], "words": heard}))
    plan = {"film": {"profile": "talking_head"}, "clips": [{"clip": i, "lines": [" ".join(f"w{i}x{k}" for k in range(6))]} for i in (1, 2, 3)],
            "angles": {"clips": [{"clip": 1}, {"clip": 2, "punch_in": True}, {"clip": 3}]}}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    out = tmp_path / "edit"
    with pytest.raises(SystemExit):
        cli.main(["edit", str(clips), "--plan", str(tmp_path / "plan.json"), "--out", str(out), "--title", "Demo", "--subtitle", "test"])
    qa = json.loads((out / "edit_qa.json").read_text())
    assert (out / "final.mp4").exists() and (out / "contact_sheet.jpg").exists()
    g = {x["id"]: x["pass"] for x in qa["gates"]}
    assert g["G8"] and g["G9"] and g["G12"] and g["G13"], qa["gates"]
    assert all(r["offset_ms"] is not None and abs(r["offset_ms"]) <= 10 for r in qa["av"]["segments"]), qa["av"]
    L = qa["loudness"]["final"]
    assert abs(L["I"] + 14) <= 1.0 and L["TP"] <= -1.95, L
    before = [c["colour_before"]["dE76_vs_anchor"] for c in qa["clips"]]
    after = [c["colour_after"]["dE76_vs_anchor"] for c in qa["clips"]]
    assert max(before) > 3 and max(after) < max(before) / 2, (before, after)
    assert after[0] < 0.3, after          # the anchor clip keeps its pixels (no implicit colour-matrix guess at the concat)
    assert qa["final"]["width"] == 320 and qa["loudness"]["final"]["TP"] <= -1.95
    assert qa["contact_sheet"]["frames"] == 42 and len(qa["contact_sheet"]["t"]) == 42


def test_edit_dry_run_writes_edl(tmp_path):
    if not HAS_FF:
        pytest.skip("needs ffmpeg")
    _clip(tmp_path / "c" / "clip1.mp4", "gray", 300)
    qa = ED.edit(tmp_path / "c", None, TH, tmp_path / "o", dry_run=True, require_wer=False)
    assert qa["dry_run"] and (tmp_path / "o" / "edl.json").exists()


def test_wer_from_word_file_feeds_editor(tmp_path):
    words = _w("one two three four")
    (tmp_path / "w.json").write_text(json.dumps([{"w": x["w"], "t0": x["t0"], "t1": x["t1"], "p": 0.9} for x in words]))
    cli.main(["wer", "-", "one two three four", "--seconds", "5", "--words", str(tmp_path / "w.json"), "--out", str(tmp_path / "clip1.wer.json")])
    r = json.loads((tmp_path / "clip1.wer.json").read_text())
    assert r["win_wer"] == [0.0] and len(r["words"]) == 4
    assert ED.find_clips(tmp_path, wer_dir=tmp_path) == []   # no video here; the lookup itself must not fail


def _clip2(path, colour="0x806050", size="320x180", fps=24, audio="noise", seed=1, d=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={colour}:s={size}:r={fps}:d={d}"]
    if audio == "noise":
        cmd += ["-f", "lavfi", "-i", f"anoisesrc=d={d}:c=pink:a=0.3:r=48000:seed={seed}", "-shortest"]
    elif audio == "silent":
        cmd += ["-f", "lavfi", "-i", f"anullsrc=r=48000:cl=stereo:d={d}", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"] + (["-c:a", "aac"] if audio != "none" else []) + [str(path)]
    subprocess.run(cmd, check=True)


@pytest.mark.skipif(not HAS_FF, reason="needs ffmpeg")
def test_mixed_rates_shapes_and_missing_audio(tmp_path):
    _clip2(tmp_path / "clip1" / "video.mp4", fps=24, seed=1)
    _clip2(tmp_path / "clip2" / "video.mp4", fps=30, seed=2, colour="0x5a4030")
    _clip2(tmp_path / "clip3" / "video.mp4", size="180x320", seed=3)
    _clip2(tmp_path / "clip4" / "video.mp4", audio="none", colour="0x907060")
    qa = ED.edit(tmp_path, None, TH, tmp_path / "o", require_wer=False)
    w = " ".join(qa.get("warnings", []))
    assert "30.000 fps" in w and "letterboxed" in w
    assert qa["av"]["stream_gap_s"] <= 1 / 24 + 0.03, qa["av"]
    assert all(r["offset_ms"] is None or abs(r["offset_ms"]) <= 10 for r in qa["av"]["segments"]), qa["av"]
    assert abs(qa["final"]["duration"] - 12.0) < 0.1


@pytest.mark.skipif(not HAS_FF, reason="needs ffmpeg")
def test_silent_film_is_not_amplified(tmp_path):
    _clip2(tmp_path / "clip1" / "video.mp4", audio="silent")
    qa = ED.edit(tmp_path, None, TH, tmp_path / "o", require_wer=False)
    assert any("near silent" in x for x in qa["warnings"]) and qa["loudness"]["tries"][-1]["gain_db"] == 0.0


def test_clip_with_only_flawed_words_is_left_out():
    heard = _w("and that is how it works")
    f = ED.find_flaws(heard, "", earlier=[x["w"] for x in heard])
    cl = ED.cut_list(heard, 3.0, f, TH)
    assert cl["segments"] == []


def test_shared_phrase_with_previous_script_is_kept():
    prev = [x["w"] for x in _w("this is the end of the first part")]
    f = ED.find_flaws(_w("at the end of the day we ship it"), "at the end of the day we ship it", earlier=prev,
                      earlier_ref="this is the end of the first part")
    assert f == []


def test_out_point_follows_the_last_kept_word():
    heard = _w("we ship it today um")
    cl = ED.cut_list(heard, 5.0, ED.find_flaws(heard, "we ship it today"), TH)
    assert len(cl["segments"]) == 1 and cl["out"] == pytest.approx(heard[3]["t1"] + 0.25)

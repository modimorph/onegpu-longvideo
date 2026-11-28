import json
from pathlib import Path

import pytest

from onegpu_longvideo import cli, profiles
from onegpu_longvideo import planner as PL

ROOT = Path(__file__).parent.parent
EXAMPLE = ROOT / "configs" / "scripts" / "example_talking_head.yaml"
TH = profiles.load("talking_head")


def _script(n_clips, seconds, lines_per_clip):
    clips = [PL.Clip(i + 1, f"c{i + 1}", seconds, [PL.Line(t, t) for t in lines_per_clip[i]]) for i in range(n_clips)]
    return PL.Script("t", clips, {"tag": "S1", "role": "the presenter"}, {})


def test_all_profiles_load_and_differ():
    names = profiles.names()
    assert {"talking_head", "tutorial", "podcast", "cinematic_short", "ad", "documentary"} <= set(names)
    loaded = {n: profiles.load(n) for n in names}
    for p in loaded.values():
        assert p["planner"]["speech"]["wps"][0] < p["planner"]["speech"]["wps"][1]
        assert p["editor"]["audio"]["lufs"] == -14 and p["editor"]["review"]["contact_sheet"] == [7, 6]
    assert loaded["tutorial"]["planner"]["angles"]["mode"] == "punch_in"
    assert loaded["ad"]["planner"]["hook"]["first_line_by_s"] == 3
    assert loaded["cinematic_short"]["editor"]["cuts"]["internal"] is False


@pytest.mark.parametrize("raw,clean,kind", [
    ("the kind you'd, you'd put in a home PC.", "The kind you'd put in a home PC.", "stutter"),
    ("the speech [cough] starts to slip.", "The speech starts to slip.", "cough"),
    ("heading for an hour... well, nearly an hour.", "Heading for an hour.", "self_correction"),
    ("so um this is it.", "So this is it.", "filler"),
    ("we stopped -- then went on.", "We stopped. Then went on.", "fake_stop"),
])
def test_fake_actions_fixed(raw, clean, kind):
    out, kinds = PL.fix_fake_actions(raw, TH["planner"]["banned_actions"])
    assert out == clean and kind in kinds


def test_repeated_sentence_fails_gate():
    s = _script(2, 10, [["One shot from one home graphics card, made in one single pass today."],
                        ["Again: one shot from one home graphics card, it is true."]])
    p = PL.plan(s, TH)
    g2 = next(g for g in p["gates"] if g["id"] == "G2")
    assert not g2["pass"] and not p["gates_pass"]


def test_density_band_and_deadline_10s():
    sp = TH["planner"]["speech"]
    assert PL.speech_window(10, sp) == (0.3, 8.3)
    assert PL.speech_window(15, sp) == (0.3, 13.3)
    assert PL.speech_window(29.96, sp)[1] == pytest.approx(28.26)
    s = _script(1, 10, [["word " * 30]])
    d = PL.density(s, TH)
    assert d["clips"][0]["band"] == [24, 25] or d["clips"][0]["band"][0] >= 22
    assert any("too_many_words" in e["type"] for e in d["errors"])


def test_angle_plan_3min_bookend():
    topics = ["graphics card render speed", "invented words problem memory", "fix tiles attention cache",
              "voice lips timing", "quality seeds review", "repo open try"]
    s = _script(6, 29.96, [[f"{t} sentence about {t}."] for t in topics])
    a = PL.angle_plan(s, TH)
    setups = [r["setup"] for r in a["clips"]]
    assert setups[0] == "A" and setups[1] == "A" and setups[-1] == "A"
    assert a["band"] == [2, 3] and 2 <= a["changes"] <= 3 and not a["errors"]
    assert all(r["picture3"].startswith("new set-up") for r in a["clips"] if r["change_from_prev"])


def test_gesture_budget_and_finger_cap():
    lines = [["So, two problems here, and two things to fix."] * 3 for _ in range(6)]
    s = _script(6, 29.96, lines)
    g = PL.gesture_plan(s, TH)
    assert g["used"].get("count_fingers", 0) <= 1
    assert sum(g["used"].get(x, 0) for x in TH["planner"]["gestures"]["finger_types"]) <= 2
    assert g["total"] <= g["max_total"] and not g["errors"]


def test_example_plan_passes_and_prompts_lint_clean(tmp_path):
    with pytest.raises(SystemExit) as e:
        cli.main(["plan", str(EXAMPLE), "--out", str(tmp_path)])
    assert e.value.code == 0
    p = json.loads((tmp_path / "plan.json").read_text())
    assert p["gates_pass"] and len(p["prompts"]) == 3
    txt = (tmp_path / "prompts" / "clip1.txt").read_text()
    assert "<d>[English]" in txt and "small wave" in txt
    assert "rests both hands" in (tmp_path / "prompts" / "clip2.txt").read_text()
    assert (tmp_path / "prompts" / "clip3.lines.txt").exists()


def test_markdown_segments_parsed(tmp_path):
    md = tmp_path / "SCRIPT.md"
    md.write_text("# demo\n\n## Segment 1 (0-10 s): Hook\n\n| # | Local time (s) | Spoken line | Gesture |\n|---|---|---|---|\n"
                  "| 1 | 0.3-5.0 | Hi, hi there [cough] friends. | wave |\n")
    s = PL.load_script(md)
    assert s.clips[0].seconds == 10 and s.clips[0].lines[0].cue == "wave"
    tp = PL.text_pass(s, TH)
    assert s.clips[0].lines[0].text == "Hi there friends." and tp["fixes"]


def test_plan_cli_exit_1_on_failed_gate(tmp_path):
    bad = tmp_path / "s.yaml"
    bad.write_text("clip_seconds: 10\nclips:\n  - lines: ['Too short.']\n")
    with pytest.raises(SystemExit) as e:
        cli.main(["plan", str(bad), "--out", str(tmp_path / "o")])
    assert e.value.code == 1


def test_llm_request_has_whole_script():
    s = PL.load_script(EXAMPLE)
    req = PL.llm_review_request(s, PL.text_pass(s, TH))
    assert "clip 1" in req and "clip 3" in req and "fake spoken action" in req


def test_fact_sheet_catches_contradicted_claims(tmp_path):
    s = _script(1, 10, [["So we render two takes and pick the best one.", "Thirty seconds takes about three minutes."]])
    s.facts = [{"id": "takes", "keywords": ["takes"], "forbid": [r"\btwo takes\b"], "note": "three clips were best of one"},
               {"id": "render", "keywords": ["thirty seconds", "takes about"], "allowed": ["ten minutes"], "note": "native 720p: 620-673 s"}]
    hits = PL.fact_check(s)
    assert {h["fact"] for h in hits} == {"takes", "render"}
    p = PL.plan(s, TH)
    assert not next(g for g in p["gates"] if g["id"] == "G2F")["pass"]
    assert "Fact sheet" in PL.llm_review_request(s, p["text_pass"])

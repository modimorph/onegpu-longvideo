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
    ("we stopped... then went on.", "We stopped. Then went on.", "fake_stop"),
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


@pytest.mark.parametrize("profile,script", [("talking_head", "example_talking_head"), ("tutorial", "example_tutorial"),
                                            ("podcast", "example_podcast"), ("cinematic_short", "example_cinematic"),
                                            ("ad", "example_ad"), ("documentary", "example_documentary")])
def test_every_profile_example_passes(profile, script):
    prof = profiles.load(profile)
    p = PL.plan(PL.load_script(ROOT / "configs" / "scripts" / f"{script}.yaml"), prof)
    assert p["gates_pass"], [g for g in p["gates"] if not g["pass"]]
    text = " ".join(x["prompt"] for x in p["prompts"])
    if profile != "talking_head":
        assert "hands on the desk" not in text, "talking-head rest wording leaked into another profile"
    for r in p["density"]["clips"]:   # in band implies on time (G3 and G4 agree)
        assert r["est_last_word_s"] <= r["deadline_s"] + 1e-6


def test_profile_replace_drops_base_gestures():
    assert "count_fingers" not in profiles.load("cinematic_short")["planner"]["gestures"]["types"]
    assert "count_fingers" in profiles.load("podcast")["planner"]["gestures"]["types"]


def test_speaker_rest_overrides_profile():
    s = _script(1, 10, [["Quiet line one here.", "Quiet line two here."]])
    s.speaker["rest"] = "keeps both hands on the steering wheel"
    assert all(r["wording"] == "keeps both hands on the steering wheel" or r["type"] != "rest" for r in PL.gesture_plan(s, TH)["lines_plan"])


def test_non_english_words_stay_whole_and_fillers_go():
    assert PL.norm_words("Größe und Qualität") == ["größe", "und", "qualität"]
    out, kinds = PL.fix_fake_actions("Äh, das ist ganz einfach.", TH["planner"]["banned_actions"], "German")
    assert out == "Das ist ganz einfach." and "filler" in kinds
    s = _script(1, 10, [["Hallo zusammen."]])
    s.speaker["language"] = "German"
    assert any(w["type"] == "speech_band_untested_language" for w in PL.density(s, TH)["warnings"])



@pytest.mark.parametrize("text", [
    "No, actually, it works on every card.", "I know that that is true.", "She had had enough of slow renders.",
    "The model works (breathing room matters).", "Check the (laughter track) setting.", "Use e.g. a still frame.",
    "Did you know, the card has enough memory?", "Very very fast.",
])
def test_normal_english_is_not_changed(text):
    assert PL.fix_fake_actions(text, TH["planner"]["banned_actions"]) == (text, [])


def test_ellipsis_keeps_the_words_after_it():
    out, _ = PL.fix_fake_actions("We waited... now we ship it today.", TH["planner"]["banned_actions"])
    assert out == "We waited. Now we ship it today."
    assert PL.fix_fake_actions("The model — trained on reels — works.", TH["planner"]["banned_actions"])[0] == "The model, trained on reels, works."
    assert PL.fix_fake_actions("Er sagte, es geht.", TH["planner"]["banned_actions"], "German") == ("Er sagte, es geht.", [])


def test_repeat_inside_one_line_and_stage_direction_lines():
    hits = PL.repeated_runs([("l1", "We made one long shot on one card, and we made one long shot on one card.")], 5)
    assert hits
    s = _script(1, 10, [["Hello there, friends of the show.", "[laughs]", "We start right now with the plan."]])
    PL.text_pass(s, TH)
    assert len(s.clips[0].lines) == 2


def test_yaml_lines_with_colons_and_booleans(tmp_path):
    y = tmp_path / "s.yaml"
    y.write_text("clip_seconds: 10\nclips:\n  - seconds: 20\n    lines:\n      - Step one: open the app\n      - yes\n")
    s = PL.load_script(y)
    assert [ln.text for ln in s.clips[0].lines] == ["Step one: open the app", "yes"] and s.clips[0].seconds == 20
    assert PL.load_script(y, clip_seconds=10).clips[0].seconds == 10      # --clip-seconds overrides per-clip seconds


def test_markdown_three_column_table(tmp_path):
    md = tmp_path / "s.md"
    md.write_text("## Segment 1 (0-10 s): A\n\n| # | t | line |\n|---|---|---|\n| 1 | 0.3-4 | First line here. |\n| 2 | 4-8 | Second line here. |\n")
    assert [ln.text for ln in PL.load_script(md).clips[0].lines] == ["First line here.", "Second line here."]


@pytest.mark.parametrize("n,sec,prof", [(12, 30, "talking_head"), (40, 30, "talking_head"), (2, 15, "ad"), (1, 400, "talking_head"),
                                         (3, 61, "talking_head"), (5, 15, "tutorial"), (7, 10, "cinematic_short")])
def test_angle_plan_scales_and_stays_in_band(n, sec, prof):
    import time

    clips = [PL.Clip(i + 1, "c", sec, [PL.Line(f"topic{i} words about topic{i} here{i}.")]) for i in range(n)]
    t = time.time()
    a = PL.angle_plan(PL.Script("t", clips), profiles.load(prof))
    assert time.time() - t < 2 and not a["errors"], (a["band"], a["changes"])
    seq = [r["setup"] for r in a["clips"]]
    assert all(not r["change_from_prev"] or seq[i] != seq[i - 1] for i, r in enumerate(a["clips"]))


def test_varied_rest_variant_extends_talking_head():
    v = profiles.load("talking_head_varied")
    assert v["planner"]["speech"]["wps"] == TH["planner"]["speech"]["wps"] and len(v["planner"]["gestures"]["rest_variants"]) == 4
    s = _script(1, 29.96, [[f"Neutral line number {i} here." for i in range(6)]])
    rests = [r["wording"] for r in PL.gesture_plan(s, v)["lines_plan"] if r["type"] == "rest"]
    assert all(a != b for a, b in zip(rests, rests[1:]))


def test_replan_with_fewer_clips_removes_stale_prompts(tmp_path):
    PL.write(PL.plan(PL.load_script(EXAMPLE), TH), tmp_path)
    (tmp_path / "prompts" / "clip9.txt").write_text("stale")
    (tmp_path / "prompts" / "notes.txt").write_text("keep me")
    PL.write(PL.plan(PL.load_script(ROOT / "configs" / "scripts" / "example_ad.yaml"), profiles.load("ad")), tmp_path)
    names = sorted(x.name for x in (tmp_path / "prompts").iterdir())
    assert names == ["clip1.lines.txt", "clip1.txt", "clip2.lines.txt", "clip2.txt", "notes.txt"]


def test_setup_reference_prompts_follow_the_plan():
    topics = ["graphics card render speed", "invented words problem memory", "fix tiles attention cache",
              "voice lips timing", "quality seeds review", "repo open try"]
    a = PL.angle_plan(_script(6, 29.96, [[f"{t} sentence about {t}."] for t in topics]), TH)
    assert set(a["setup_refs"]) == {"B", "C"} and "three-quarter" in a["setup_refs"]["B"] and "{side}" not in a["setup_refs"]["B"]
    assert "torso is clearly angled" in next(r["text"] for r in a["clips"] if r["setup"] == "B")


def test_calibrate_reports_measured_rate(tmp_path):
    p = PL.plan(PL.load_script(EXAMPLE), TH)
    for n in (1, 2):   # 24 words spread over 9.8 s: 2.45 w/s, past the 8.3 s deadline
        words = [{"w": f"w{k}", "t0": round(k * 0.408, 3), "t1": round(k * 0.408 + 0.3, 3)} for k in range(24)]
        (tmp_path / f"clip{n}.json").write_text(json.dumps(words))
    cal = PL.calibrate(p, tmp_path, TH)
    assert cal["median_rate_wps"] == pytest.approx(2.45, abs=0.05) and cal["clips_past_deadline"] == [1, 2]
    y = tmp_path / "cal.yaml"
    y.write_text(PL.calibration_profile(cal, "talking_head"))
    prof = profiles.load(str(y))
    assert prof["planner"]["speech"]["wps"][1] < 2.8 and prof["planner"]["gestures"]["rest"].startswith("rests both hands")


def test_setup_cycle_never_repeats_and_bookends_when_possible():
    for seq in (["A", "B", "C", "A"], ["A", "B", "A", "B"], ["A", "B"]):
        for n in range(7):
            for book in (True, False):
                r = PL._run_setups(seq, n, book)
                assert len(r) == n + 1 and all(a != b for a, b in zip(r, r[1:]))
    assert PL._run_setups(["A", "B", "C", "A"], 3, True) == ["A", "B", "C", "A"]


def test_planner_fuzz_never_raises_and_keeps_angle_band():
    import random

    random.seed(3)
    vocab = ("the model card speech clip plan camera colour voice two three only then now so but here look remember years "
             "you get quiet ... — [cough] um, you'd, you'd größe café 42 3.5 % : ( ) don't").split()
    for _ in range(200):
        prof = profiles.load(random.choice(profiles.names()))
        clips = [PL.Clip(i + 1, "c", random.choice([3, 7.5, 10, 15, 16, 29.96, 61]),
                         [PL.Line(t, t) for t in [" ".join(random.choice(vocab) for _ in range(random.randint(0, 20)))
                                                  for _ in range(random.randint(0, 5))]]) for i in range(random.randint(1, 20))]
        p = PL.plan(PL.Script("t", clips, {"language": random.choice(["English", "German"])}, {}), prof)
        assert not p["angles"]["errors"]


def test_llm_review_speaks_openai_chat_format(tmp_path, monkeypatch):
    import http.server
    import threading

    seen = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["auth"] = self.headers.get("Authorization")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            out = json.dumps({"choices": [{"message": {"content": "clip 1 line 2: repeated idea -> cut it"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("OLV_LLM_URL", f"http://127.0.0.1:{srv.server_port}/v1")
    monkeypatch.setenv("OLV_LLM_MODEL", "writer")
    monkeypatch.setenv("OLV_LLM_KEY", "test-key")
    with pytest.raises(SystemExit):
        cli.main(["plan", str(EXAMPLE), "--llm", "--out", str(tmp_path)])
    srv.shutdown()
    assert seen["path"] == "/v1/chat/completions" and seen["auth"] == "Bearer test-key" and seen["body"]["model"] == "writer"
    assert "Script:" in seen["body"]["messages"][0]["content"]
    assert (tmp_path / "text_review.md").read_text().startswith("clip 1 line 2")


def test_punch_profile_gives_each_setup_its_own_crop():
    topics = ["graphics card render speed", "invented words problem memory", "fix tiles attention cache",
              "voice lips timing", "quality seeds review", "repo open try"]
    a = PL.angle_plan(_script(6, 29.96, [[f"{t} sentence about {t}."] for t in topics]), profiles.load("talking_head_punch"))
    scales = {r["setup"]: r["punch_scale"] for r in a["clips"]}
    assert scales["A"] is None and scales["B"] != scales["C"] and not a["setup_refs"]
    assert all(r["picture3"] != "new set-up: no last-frame chaining; the set-up reference still is the framing picture" for r in a["clips"])



def test_seed_only_chain_uses_one_picture():
    prof = profiles.load("talking_head_punch")
    s = PL.load_script(EXAMPLE)
    p = PL.plan(s, prof)
    assert p["gates_pass"]
    first, second = p["prompts"][0]["prompt"], p["prompts"][1]["prompt"]
    assert "<Picture 2>" in first and "<Picture 2>" not in second and "<Picture 1> is the last moment of the previous shot" in second
    assert p["prompts"][1]["pictures"] and p["prompts"][0]["pictures"] is None
    assert "chest up" in first and "chest up" not in second and "no zoom" in second   # framing comes from the picture



def test_talking_head_chains_with_one_picture_but_other_profiles_do_not():
    assert profiles.load("talking_head")["planner"]["prompt"]["chain_refs"] == "seed_only"
    assert profiles.load("talking_head_varied")["planner"]["prompt"]["chain_refs"] == "seed_only"
    assert profiles.load("documentary")["planner"]["prompt"]["chain_refs"] == "all"

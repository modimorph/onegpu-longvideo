import json

import pytest

from onegpu_longvideo import cli, flow
from onegpu_longvideo import recipe as R

GOOD = {
    "render_s": 635, "upscale_s": 20, "grade": True,
    "raw": {"width": 1280, "height": 720, "frames": 719},
    "final": {"width": 1920, "height": 1080, "vcodec": "h264", "crf": 18, "acodec": "aac", "audio_kbps": 192},
    "qa": {"stage1_pass": True, "flagged_reviewed": True, "speaker_match": 0.7, "lip_sync_ok": True, "contact_sheet_reviewed": True},
    "edit": {"repeats_cut": True},
    "voice": {"used": True, "len_diff": 0, "wer_guard": True, "listening_approved": True},
}
WER = {"win_wer": [0, 0, 0, 0, 0, 0], "last_word_s": 28.1}


def _run(tmp_path, manifest=GOOD, wer=WER, plan_ok=True, edit_ok=True):
    (tmp_path / "flow_run.json").write_text(json.dumps(manifest))
    (tmp_path / "plan.json").write_text(json.dumps({"gates_pass": plan_ok, "gates": [{"id": "G3", "pass": plan_ok}]}))
    (tmp_path / "edit_qa.json").write_text(json.dumps({"gates_pass": edit_ok, "gates": [{"id": "G11", "pass": edit_ok}]}))
    (tmp_path / "wer_raw.json").write_text(json.dumps(wer))
    (tmp_path / "wer_final.json").write_text(json.dumps(wer))
    return tmp_path


def _fails(rows):
    return {r["gate"] for r in rows if not r["pass"]}


def test_good_run_passes(tmp_path):
    assert flow.passed(flow.check(_run(tmp_path)))


def test_wer_window_fails(tmp_path):
    rows = flow.check(_run(tmp_path, wer={"win_wer": [0, 0, 0.07, 0, 0, 0], "last_word_s": 28.1}))
    assert "0 % WER in every 5 s window (raw take)" in _fails(rows)


def test_late_last_word_fails(tmp_path):
    assert "last word by 28.3 s" in _fails(flow.check(_run(tmp_path, wer={**WER, "last_word_s": 29.5})))


def test_frames_and_encode_fail(tmp_path):
    bad = {**GOOD, "raw": {"width": 768, "height": 432, "frames": 721}, "final": {**GOOD["final"], "crf": 23}}
    f = _fails(flow.check(_run(tmp_path, bad)))
    assert {"719 frames", "raw 1280x720", "encode h264 CRF 18, AAC 192k"} <= f


def test_voice_needs_human_yes(tmp_path):
    bad = {**GOOD, "voice": {**GOOD["voice"], "listening_approved": False}}
    assert "human listening check approved" in _fails(flow.check(_run(tmp_path, bad)))


def test_voice_step_skipped_when_unused(tmp_path):
    rows = flow.check(_run(tmp_path, {**GOOD, "voice": {"used": False}}))
    assert not [r for r in rows if r["step"] == 10]


def test_cli_exit_codes(tmp_path):
    with pytest.raises(SystemExit) as e:
        cli.main(["flow-check", str(_run(tmp_path))])
    assert e.value.code == 0
    with pytest.raises(SystemExit) as e:
        cli.main(["flow-check", str(_run(tmp_path, wer={"win_wer": [0.1], "last_word_s": 1}))])
    assert e.value.code == 1


def test_flow_default_recipe_matches_flow_settings():
    from pathlib import Path

    r = R.load(Path(__file__).parent.parent / "configs" / "flow.default.yaml")
    assert (r.width, r.height, r.frames, r.seed, r.steps, r.sampler, r.guidance) == (1280, 720, 719, 7329, 8, "euler", 1.0)
    assert (r.identity_strength, r.realism, r.attention, r.vsa_keep_percent) == (0.8, 0.3, "vsa", 10)


def test_planner_and_editor_gates_required(tmp_path):
    f = _fails(flow.check(_run(tmp_path, plan_ok=False, edit_ok=False)))
    assert {"planner gates passed (plan.json)", "editor gates passed (edit_qa.json)"} <= f
    (tmp_path / "plan.json").unlink()
    assert "planner gates passed (plan.json)" in _fails(flow.check(tmp_path))


def test_short_clip_gates_follow_seconds(tmp_path):
    m = {**GOOD, "seconds": 10, "raw": {"width": 1280, "height": 720, "frames": 243}}
    rows = flow.check(_run(tmp_path, m, wer={"win_wer": [0, 0], "last_word_s": 8.2}))
    assert flow.passed(rows), _fails(rows)
    assert "last word by 8.34 s" in _fails(flow.check(_run(tmp_path, m, wer={"win_wer": [0, 0], "last_word_s": 9.0})))



def test_film_mode_reads_every_clip_from_edit_qa(tmp_path):
    m = {**GOOD, "film": True, "seconds": 10, "raw": {"width": 1280, "height": 720, "frames": 243}}
    _run(tmp_path, m)
    (tmp_path / "wer_final.json").unlink()
    (tmp_path / "plan.json").write_text(json.dumps({"gates_pass": True, "gates": [], "clips": [{"clip": 1, "seconds": 10}, {"clip": 2, "seconds": 10}]}))
    wc = [{"clip": 1, "checked": True, "win_wer": [0, 0], "last_word_s": 8.1}, {"clip": 2, "checked": True, "win_wer": [0, 0.07], "last_word_s": 9.8}]
    (tmp_path / "edit_qa.json").write_text(json.dumps({"gates_pass": True, "gates": [], "word_check": {"clips": wc}}))
    f = _fails(flow.check(tmp_path))
    assert f == {"every clip: last word by S - 1.66 s", "every clip: 0 % WER in every 5 s window"}
    wc[1].update(win_wer=[0, 0], last_word_s=8.2)
    (tmp_path / "edit_qa.json").write_text(json.dumps({"gates_pass": True, "gates": [], "word_check": {"clips": wc}}))
    assert flow.passed(flow.check(tmp_path))
